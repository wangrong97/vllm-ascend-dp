# DSA Triton 算子 P 矩阵 HiF8 量化修复分析

> 涉及文件：`vllm_ascend/attention/kv_quant_sparse_attn_triton.py`
> 算子：`_dsa_decode_kernel`（DSA decode 路径，`cmp_ratio` ∈ {1, 4, 128}）

---

## 1. 概述

| | 内容 |
|---|---|
| **现象** | 启用 `enable_dsa_triton_decode` 后，triton DSA 算子对 P 矩阵（softmax 指数项）做 HiF8 量化，端到端对话结果全部错误 |
| **根因** | HiF8 被放在 **online softmax 循环体内、每 tile 对 `exp(score - running_max)` 量化**。HiF8 是非线性量化，与 online softmax 的 running-max rescale **不可交换**，叠加了一份额外的量化噪声 |
| **修复** | 改为 **两遍扫描全局 HiF8**：Pass 1 先求全局 rowmax `m_final`，Pass 2 用固定 `m_final` 对 `exp(score - m_final)` 一次性量化加权。与 pytorch 参考实现语义完全对齐 |
| **验证** | NPU 实测：诊断脚本 mean 误差 3.4e-3 → 2.2e-3；端到端 **aime2024 准确率 66.67%**（修复前全部错误） |

---

## 2. 背景：算子流程与业务需求

DeepSeek-V4 A8C4 量化路径中，DSA（Dynamic Sparse Attention）decode 的两套实现：

- **ascendc 原生算子**：标准 FlashAttention（`SoftmaxFlashV2 + Cast(fp32→bf16)`），fp32 累积，**P 矩阵不做 HiF8**。
- **triton 算子**（本文件）：业务要求**额外对 P 矩阵做 HiF8 量化**（模拟 A8C4 中间结果用 HiF8 存储）。

两者的唯一流程差异就是 triton 多了「P 矩阵 HiF8」这一步。Q 在外部已做 HiF8（per-tensor, scale=8.0）、KV 在外部已做 FP4（per-block）量化——这部分两者一致。

因此问题归结为：**triton 的 P 矩阵 HiF8 实现方式是否正确。**

---

## 3. 根因：per-tile HiF8 与 online softmax 不可交换

### 3.1 标准 online softmax 为何精确

无量化时，online softmax 靠 `alpha = exp(m_i - m_new)` 把"旧 scale 的累积"换算到"新 scale"，依赖 exp 的恒等式：

```
exp(s - m_new) = exp(s - m_i) * exp(m_i - m_new) = exp(s - m_i) * alpha
```

因为 exp **线性可缩放**，"先按 m_i 算、再 rescale 到 m_new"与"直接按 m_new 算"**完全等价**——这就是单遍 online softmax 能精确等于全局 softmax 的数学基础。

### 3.2 HiF8 破坏了可交换性

HiF8 是**非线性**的 round-to-grid 量化 `Q(x) = round(x/s) * s`。对任意缩放因子 `k ≠ 1`：

```
Q(a) * k  ≠  Q(a * k)        ← 非线性，不可交换
```

### 3.3 落到 per-tile vs global 的差异

设某 token j 的最终全局 max 为 `m*`，但它所在 tile 处理时 running max 只有 `m_T < m*`。令：

- `a = exp(s_j - m_T)`：tile 时刻的值，**偏大**
- `k = exp(m_T - m*) < 1`：后续 rescale 因子
- `a * k = exp(s_j - m*)`：最终正确值

| | 该 token 最终贡献 | 在什么量级上做 round |
|---|---|---|
| **per-tile（改前）** | `Q(a) * k` | 在**偏大的 `a`** 上 round，再缩小 |
| **global（改后/参考）** | `Q(a * k)` | 在**正确的最终值 `a*k`** 上 round |

`a` 比 `a*k` 大 `1/k` 倍，落在 HiF8 更大量级的指数带（mantissa bits 更少、grid 步长更大），round 的绝对偏差更大，缩小后偏差被保留 → **每 tile 量化叠加了一份本不该有的量化噪声**。

### 3.4 诊断脚本实测（NPU，`diag_triton_vs_ref.py`）

```
triton(per-tile hif8) vs ref(global hif8)：mean 3.4e-3
triton(global hif8)   vs ref(global hif8)：mean 2.2e-3   ← 改后
```

per-tile 比 global **多 ~55% 的 mean 误差**。更关键的是：

```
triton:True  vs triton:False  = 2.44e-2   (改前 per-tile)
ref:True     vs ref:False     = 2.40e-2
```

改前两者不持平（triton 的 HiF8 总影响偏大）；改后两者完全持平，证明 **triton 的 HiF8 行为已与参考实现一致**。

这份额外误差在几十层 attention 里逐层放大，就是"对话全部错误"的根因。同时确认：**attention 逻辑本身（fp8/e8m0 dequant、paged gather、online softmax 循环）是正确的**——`triton:False vs ref:False` 的 mean 仅 5.6e-4（且主要来自参考实现用 bf16 matmul、triton 用 fp32 点积的精度差，triton 反而更精确）。

---

## 4. 修复方案：两遍扫描全局 HiF8

```
Pass 1（只求 max，不累积）:
    m_final = sink                          # 用 sink 做 seed（参考实现是 joint max(sink, scores)）
    for each KV tile:
        m_final = max(m_final, max(score_tile))
    # 结束：m_final = max(sink, 全部 score) = 全局 rowmax

Pass 2（用固定 m_final 量化+加权，无需 rescale）:
    l_i       = exp(sink - m_final)         # sink 项 seed，不 HiF8（同参考）
    acc_nope  = 0;  acc_rope = 0
    for each KV tile:
        p = HiF8( exp(score - m_final) )    # ← 所有 tile 共用同一个 m_final
        p = where(valid, p, 0)
        l_i      += sum(p)                  # ← 直接累加，没有 alpha
        acc_nope += sum(p * v_nope)
        acc_rope += sum(p * v_rope)
    out = acc / l_i
```

**核心**：Pass 2 里每个 tile 都用同一个 `m_final`，`exp(score - m_final)` 天然在同一 scale 上，`l_i` 和 `acc` 直接相加即可——**不再需要 `alpha` rescale**，从而消除了"rescale 与量化不可交换"的误差源。

这与参考实现 `_flash_attention_with_sink`（先 materialize 完整 `exp(score-rowmax)` 再一次性 HiF8）在数学上**逐 tile 等价**，只是把"materialize 整个矩阵"换成了"两遍流式扫描"。

---

## 5. 代码实现解析

### 5.1 两个 triton helper（消除三处重复）

Pass 1、Pass 2、单遍分支都要做"load KV + fp8/e8m0 dequant + 算 Q@K^T"。若内联，ori/cmp4/cmp128 × 3 处 ≈ 250 行重复。故抽取：

- **`_tile_score`**（line 100-148）：ori 的 SWA 连续 tile 与 cmp128 的 dense tile 共用。参数化 `j_start / valid_lo / valid_hi` 及指针/stride/`BLOCK_SIZE`。返回 `(score[N], nope_dequant[N,NOPE], rope[N,ROPE], valid[N])`。
- **`_cmp4_token_score`**（line 151-189）：cmp4 的单 token（标量 score）。返回 `(score, nope_dequant[NOPE], rope[ROPE], valid)`。

invalid 位置在 helper 内被 `where(valid, score, -inf)`（line 147）置 `-inf`，`max` 自动忽略。

### 5.2 Pass 1：求联合 rowmax（line 271-300）

```python
m_final = sink                                    # 272: seed = sink
for t in range(MAX_ORI_TILES):                    # 273: SWA phase
    score, _, _, _ = _tile_score(...)             # 274: 只取 score
    m_final = tl.maximum(m_final, tl.max(score))  # 280: 追全局 max
if CMP_RATIO == 4:                                # 281: sparse cmp
    for k: score, _, _, _ = _cmp4_token_score(...)
    m_final = tl.maximum(m_final, score)          # 289: 标量 max
elif CMP_RATIO == 128:                            # 290: dense cmp
    for t: score, _, _, _ = _tile_score(...)
    m_final = tl.maximum(m_final, tl.max(score))  # 300
```

Pass 1 用 `score, _, _, _` 丢弃 dequant/rope/valid，只留 score 求 max。

### 5.3 Pass 2：固定 m_final 量化加权（line 301-347）

```python
l_i = tl.exp(sink - m_final)                              # 305: sink 项 seed（不 HiF8）
acc_nope = tl.zeros([NOPE_DIM]); acc_rope = ...           # 306-307
for t in range(MAX_ORI_TILES):                            # 308
    score, nope_dequant, rope, valid = _tile_score(...)   # 309: 这次取全部
    p = _hif8_quant(tl.exp(score - m_final), HIF8_SCALE)  # 315: 固定 m_final + HiF8
    p = tl.where(valid, p, 0.0)                           # 316
    l_i      = l_i + tl.sum(p, axis=0)                    # 317: 直接加，无 alpha
    acc_nope = acc_nope + tl.sum(p[:,None]*nope_dequant)  # 318
    acc_rope = acc_rope + tl.sum(p[:,None]*rope)          # 319
# cmp4 (320-332)、cmp128 (333-347) 结构相同
```

### 5.4 改前 vs 改后逐行对照（以 ori phase 为例）

```python
# 改前（单遍 online，per-tile HiF8）：
m_new  = tl.maximum(m_i, m_block)
alpha  = tl.exp(m_i - m_new)           # ← 改后 Pass2 删除
p      = tl.exp(score - m_new)         # ← 改后用 m_final
p      = _hif8_quant(p, HIF8_SCALE)
l_i    = l_i * alpha + tl.sum(p)       # ← 改后：l_i = l_i + tl.sum(p)
acc    = acc * alpha + ...             # ← 改后：acc = acc + ...
m_i    = m_new                         # ← 改后删除（不再追 running max）

# 改后（两遍 global HiF8，Pass 2）：
p   = _hif8_quant(tl.exp(score - m_final), HIF8_SCALE)
l_i = l_i + tl.sum(p)
acc = acc + tl.sum(p * v)
```

---

## 6. 三个关键设计点

### 6.1 sink 项不 HiF8（line 305）

参考实现中 `denom = Σ HiF8(exp(s-m)) + exp(sink - m)`，**sink 项不被 HiF8**。所以 Pass 2 用 `l_i = exp(sink - m_final)` 做 seed（而非 `HiF8(exp(sink-m_final))`），保证 sink 作为 softmax 分母里"虚拟 logit"的语义与参考一致。

### 6.2 Pass 2 去掉 alpha（line 317 / 330 / 345）

这是整个修复的精髓。online softmax 的 `alpha` 唯一作用是"把旧 max 尺度的累积换算到新 max 尺度"。一旦 Pass 2 用固定 `m_final`，所有 `exp(score - m_final)` 本就在同一尺度，`l_i`、`acc` 直接累加即可——**既简化代码，又消除"rescale 与量化不可交换"的误差源**。

### 6.3 ENABLE_HIF8=False 保留单遍（line 348-405）

无量化时单遍 online softmax 本就精确，不必付两遍代价。因此 `if ENABLE_HIF8` 分两路：开启时两遍 global HiF8，关闭时单遍 online（精确）。诊断时可用 `VLLM_ASCEND_DSA_TRITON_P_HIF8=0` 切换以二分定位。

---

## 7. 代价与取舍

| 维度 | 影响 |
|---|---|
| **显存** | 无增加。仍是 per-`(q_tok, head)` 一个 program，不 materialize 整个 score 矩阵 |
| **计算** | 每个 KV tile 被 **load + dequant 两次**（Pass 1 + Pass 2），attention 部分约 2× 开销 |
| **性能可接受性** | decode 路径 attention 非瓶颈（MoE/Linear 是）；实测 aime2024 正常跑完、MTP acceptance 99%，性能可接受 |
| **精度收益** | mean 误差 3.4e-3 → 2.2e-3（消除 per-tile 额外偏差）；端到端从"全部错误"→ aime2024 66.67% |

---

## 8. 验证结果

### 8.1 算子级诊断（`diag_triton_vs_ref.py`，NPU 设备 4）

| 对比 | cmp_ratio=1 | 含义 |
|---|---|---|
| `triton:False vs ref:False` | mean 5.6e-4 | attention 逻辑正确（差异主要来自 bf16 vs fp32 精度） |
| `triton:True vs ref:True` | mean **2.2e-3**（改前 3.4e-3） | triton 与参考的 HiF8 应用方式已对齐 |
| `triton:True vs triton:False` | 2.44e-2 | HiF8 总影响（改后 == `ref:True vs ref:False` 2.4e-2，完全持平）|

### 8.2 端到端（`vllm_start_dp.sh` + `aisbench.sh`）

- smoke test（5 类问题：常识/古诗/数学/化学/方向）→ **全对**
- MTP speculative decode acceptance rate **99%**（`tokens_per_seq=2` 路径正常）
- **aime2024 accuracy = 66.67%（20/30）**——DeepSeek-V4-Flash 这一级别模型的合理优秀成绩

以上均为 **P 矩阵 HiF8 开启**（`ENABLE_HIF8` 默认 True）的结果。

---

## 9. 配置开关（环境变量）

修复同时新增两个环境变量（默认保持生产行为）：

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_ASCEND_DSA_TRITON_P_HIF8` | `1` | `1`=开启 P 矩阵 HiF8（生产 A8C4 行为）；`0`=旁路（诊断二分用） |
| `VLLM_ASCEND_DSA_TRITON_HIF8_SCALE` | `8.0` | HiF8 per-tensor scale。调小可减小固有误差（scale=8.0→1.0，固有 mean 误差 5.0e-3→2.9e-3），但属于业务方案调整 |

**HiF8 固有误差说明**：即使应用方式正确（global），HiF8 对 `p ∈ (0,1]` 用 scale=8.0 仍只有 ~12% 相对精度（`p/8 ∈ (0,0.125]` 落在 HiF8 低精度指数带）。这部分属于 A8C4 业务方案本身，无法在算子层面消除，只能调 scale。当前 scale=8.0 下模型已能正常工作（aime2024 66.67%）。

---

## 10. 附录：诊断工具

- `diag_triton_vs_ref.py`：对比 triton 算子 vs pytorch 参考实现（完整 fp8 路径），分别开关 HiF8，覆盖 cmp_ratio=1/4/128。
- `diag_scale_sweep.py`：sweep HiF8 scale，观察固有误差随 scale 变化。

运行（需先停掉占用设备的 vllm）：

```bash
cd /home/w00608002/dsk-quant/vllm-ascend-dp
ASCEND_RT_VISIBLE_DEVICES=4 DIAG_DEVICE=npu:0 python diag_triton_vs_ref.py
ASCEND_RT_VISIBLE_DEVICES=4 DIAG_DEVICE=npu:0 python diag_scale_sweep.py
```
