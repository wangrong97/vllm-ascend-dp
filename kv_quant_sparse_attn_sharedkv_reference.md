# `kv_quant_sparse_attn_sharedkv` 算子解析与 PyTorch 参考实现

## 1. 算子背景

`kv_quant_sparse_attn_sharedkv` 是 vLLM-Ascend 中 DSA（Dynamic Sparse Attention）v1 后端的**核心 attention 算子**，C++ 源码位于：

```text
csrc/attention/kv_quant_sparse_attn_sharedkv/
├── op_host/         # 算子形状推导、参数校验、tiling（host 侧）
└── op_kernel/arch35/# AIC(Cube)+AIV(Vector) 混合核实现（Ascend 950 / arch35）
```

该算子对应 `vllm_ascend/attention/dsa_v1.py` 中通过 `DeviceOperator.get_dsa_sparse_attn_op()` 获取的 `torch.ops._C_ascend.kv_quant_sparse_attn_sharedkv`，支撑 DeepSeek-V4 DSA 注意力。

算子支持三种计算模式（由属性 `cmp_ratio` 选择）：

| 模板 | `cmp_ratio` | 场景 | 说明 |
|---|---|---|---|
| `SWA_TEMPLATE_MODE` | `1` | Sliding Window Attention | 仅传入 `ori_kv` |
| `CFA_TEMPLATE_MODE` | `128` | Compressed Attention | `ori_kv` + 全量 `cmp_kv`（right-down causal） |
| `SCFA_TEMPLATE_MODE` | `4` | Sparse Compressed Attention | `ori_kv` + `cmp_kv` + `cmp_sparse_indices`（逐 query top-k） |

算子内部采用 **AIC（Cube）+ AIV（Vector）混合核**：AIC 负责 `Q@K^T` 与 `P@V` 两个 Bmm，AIV 负责 KV 读取/反量化、Softmax（Flash 形式）与多轮结果合并。两者通过三级 ping-pong buffer 流水线隐藏延迟。

---

## 2. 输入/输出与数据布局

### 2.1 输入参数

| 参数 | 输入/输出 | 描述 | 数据类型 | 数据格式 / shape |
|---|---|---|---|---|
| `q` | 输入 | Query，`layout_q=TND` 时 `[T, N1, D]`，`D=512` | BF16 | ND |
| `ori_kv` | 可选输入 | 原始（未压缩）KV cache，PageAttention 布局 `[block_num1, block_size1, KV_N, D_pack]`，`KV_N=1`、`D_pack=640` | FP8_E4M3FN（packed） | PA_ND |
| `cmp_kv` | 可选输入 | 压缩 KV cache，布局同 `ori_kv` | FP8_E4M3FN（packed） | PA_ND |
| `cmp_sparse_indices` | 可选输入 | **逐 query 的压缩 KV 逻辑 token 索引**，`layout_q=TND` 时 `[T, KV_N, K]`，`K=512/1024`（当前 `index_topk=512`） | INT32 | ND |
| `ori_block_table` | 可选输入 | `ori_kv` 的 PageAttention block 映射表 `[B, max_blocks]` | INT32 | ND |
| `cmp_block_table` | 可选输入 | `cmp_kv` 的 PageAttention block 映射表 `[B, max_blocks]` | INT32 | ND |
| `cu_seqlens_q` | 可选输入 | q 的累积 token 数 `[B+1]`（仅 TND） | INT32 | ND |
| `seqused_kv` | 可选输入 | 每个 batch 的 `ori_kv` 有效长度 `[B]` | INT32 | ND |
| `sinks` | 可选输入 | **可学习 attention sink logit**，shape `[N1]`（逐 head 标量） | FP32 | ND |
| `metadata` | 可选输入 | AICPU 元算子 `npu_kv_quant_sparse_attn_sharedkv_metadata` 的分核结果，shape 固定 `[1024]` | INT32 | ND |
| `softmax_scale` | 属性 | Attention 缩放系数（必传） | FP32 | scalar |
| `cmp_ratio` | 属性 | 压缩率，`1` / `4` / `128` | INT32 | scalar |
| `ori_mask_mode` | 属性 | `ori_kv` mask 模式，固定 `4`（band/sliding window） | INT32 | scalar |
| `cmp_mask_mode` | 属性 | `cmp_kv` mask 模式，固定 `3`（right-down causal） | INT32 | scalar |
| `ori_win_left` | 属性 | SWA 左窗，固定 `127` | INT32 | scalar |
| `ori_win_right` | 属性 | SWA 右窗，固定 `0` | INT32 | scalar |
| `tile_size` | 属性 | nope 反量化粒度，固定 `64` | INT32 | scalar |
| `rope_head_dim` | 属性 | RoPE 维度，固定 `64` | INT32 | scalar |
| `layout_q` / `layout_kv` | 属性 | 入参布局，`TND` / `PA_ND` | STRING | scalar |

### 2.2 KV 数据类型与反量化是强制的（不接受 bf16）

**这个算子一定会对 KV 做反量化，KV 不可能是 bf16。** 这由 host 侧 check 在编译期强制约束（`kv_quant_sparse_attn_sharedkv_check_single_para.cpp`），不是可选路径：

```cpp
// DTYPE_SUPPORT_MAP：ori_kv / cmp_kv 只允许这两种字节型存储类型
{ORI_KV_NAME, {ge::DT_INT8, ge::DT_FLOAT8_E4M3FN}},
{CMP_KV_NAME, {ge::DT_INT8, ge::DT_FLOAT8_E4M3FN}},

// CheckSingleParaKey：KV 的 D 维必须是 640（打包后的字节数）
OP_CHECK_IF(dSizeOriKvInput_ != 640, ... "Dimension of OriKv only support 640");

// CheckFeatureAntiquantAttr：反量化模式强制开启
OP_CHECK_IF(*opParamInfo_.kvQuantMode != 1, ...);   // kv_quant_mode 必须 = 1
OP_CHECK_IF(*opParamInfo_.tileSize != 64, ...);     // tile_size 必须 = 64
```

要点：

* KV 存储类型只接受 **INT8 或 FLOAT8_E4M3FN**（CANN 中 fp8_e4m3 常以 int8 字节承载）；二者字节布局完全相同，kernel 反量化路径一致。**传入 bf16 会在编译期 `GRAPH_FAILED`。**
* KV 的最后一维必须是 **640 字节**的打包布局（rope + nope + scale + pad），本身就内嵌了 fp8 数据与 per-tile scale。kernel 的 `DequantKv` 无条件执行 fp8→bf16 反量化，再用于 `Q@K^T`。
* 因此参考实现的 `dequant_packed_kv` 是必需步骤，与 kernel 一一对应。
* Python 侧用 `uint8` 视图承载该 buffer（`torch` 无直接 INT8-packed-as-fp8 的便捷 dtype），数值上与算子的 INT8/FP8_E4M3FN 单字节存储等价；`view(torch.float8_e4m3fn)` / `view(torch.float8_e8m0fnu)` 再 cast 即可还原。

#### 2.2.1 实际运行时：A5（Ascend950 / arch35）传入 op 的 KV cache 格式

模型本身产出的是 bf16 KV，但写入 cache 时已被量化压缩，attention op 读到的是 640B fp8-packed buffer：

1. **分配阶段**（`vllm_ascend/models/deepseek_v4.py`，`AscendDeepseekV4SWACache.get_kv_cache_spec`）：

   ```python
   if get_ascend_device_type() in {AscendDeviceType.A5}:
       self.dtype = torch.float8_e4m3fn
       vllm_config.cache_config.cache_dtype = "float8_e4m3fn"
   cached_head_size = self.head_dim + 128 if A5 else self.head_dim  # A5: 512+128=640
   return AscendSlidingWindowMLASpec(..., head_size=cached_head_size, dtype=self.dtype, ...)
   ```

   即 A5 上 `swa_kv_cache` / `compress_kv_cache` 的 dtype 是 **`torch.float8_e4m3fn`**，最后一维是 **640**（`head_dim 512 + 128`，正好等于打包字节数）。非 A5 设备则分配为 `int8`、`cached_head_size = 512`（不同打包方式）。

2. **写入阶段**（`vllm_ascend/device/device_op.py`，`A5DeviceAdaptor.dsa_kv_compress_scatter`）：

   ```python
   # Input x is unquantized bf16; cache shape is [..., head_dim(=640)].
   torch.ops._C_ascend.kv_compress_epilog(
       kv_compress_cache=cache.view(-1, 1, cache.shape[-1]),
       x=x.view(-1, x.shape[-1]),
       slot_mapping=slot_mapping, quant_group_size=64, quant_mode=2,
       round_scale_flag=True, layout=1,
   )
   ```

   `kv_compress_epilog` **融合**了"per-token/per-tile 量化 + 打包成 640B（rope bf16 + nope fp8_e4m3 + scale fp8_e8m0 + pad）+ scatter 写入 cache"全流程。模型算出的 bf16 KV 在这里被打包量化后落盘。

3. **读取阶段**：`dsa_v1.py` 把 `swa_kv_cache`（`float8_e4m3fn`、`[..., 640]`）作为 `ori_kv` 直接传给 `kv_quant_sparse_attn_sharedkv`，kernel `DequantKv` 反量化后用于 attention。

> 结论：在 A5 机器上，传入 `attn_op` 的 `ori_kv`/`cmp_kv` 是 **dtype=`float8_e4m3fn`、shape=`[num_blocks, block_size, 1, 640]`** 的 fp8 打包张量（不是 bf16）。参考实现用 `uint8` 接收并 `view(torch.float8_e4m3fn)`，与之等价。

### 2.3 KV Packed 格式（640 字节）

`ori_kv` / `cmp_kv` 每个 token 打包为 **640 字节**，按序排列（见 README 约束与 kernel `DequantKv`）：

| 区域 | 含义 | 精度 | 字节数 |
|---|---|---|---|
| `kv_rope` | RoPE 部分 | bf16 | `64 × 2 = 128` |
| `kv_nope` | 非 RoPE 部分 | fp8_e4m3fn | `448 × 1 = 448` |
| `nope_quant_scale` | nope 的 per-tile（64 维）反量化 scale | fp8_e8m0fnu | `448 / 64 = 7` |
| `pad` | 尾部对齐 padding | — | `640 - (128+448+7) = 57` |

反量化流程：先对 `kv_nope` 按 64 维 tile 用对应 `fp8_e8m0` scale 还原到 bf16，再与 `kv_rope` 拼接。**kernel 内部输出布局为 `[nope(448) | rope(64)]`（即 nope 在前、rope 在后）**，得到有效维度 `D=512` 的 KV。

> **Shared KV**：本算子中 `K` 和 `V` 共享同一份反量化结果（`K̃ = Ṽ`），反量化后的 `[nope | rope]` 张量同时作为 Key 和 Value。

### 2.4 输出

| 参数 | 描述 | 数据类型 | 数据格式 / shape |
|---|---|---|---|
| `attention_out` | Attention 输出，`layout_q=TND` 时 `[T, N1, D]` | BF16 | ND |
| `softmax_lse` | Softmax 统计值，**当前输出为无效值**（`return_softmax_lse` 预留未支持） | FP32 | ND |

---

## 3. 算子内部流程

### 3.1 高层算法

$$
O = \text{softmax}\!\left(Q@\tilde{K}^{\top}\cdot \text{softmax\_scale},\ \text{sink}\right)@\tilde{V},\qquad \tilde{K}=\tilde{V}
$$

其中参与计算的 KV `K̃ = Ṽ` 由以下部分按 `cmp_ratio` 按需拼接：

1. **`ori_kv`**：Sliding window 范围内的原始 KV（band mask）。
2. **`cmp_kv`**（`cmp_ratio=128`）：从位置 `0` 到 `s2LineEndIdx / 128` 的全量压缩 KV（right-down causal）。
3. **`cmp_kv` + `cmp_sparse_indices`**（`cmp_ratio=4`）：每个 query token 各自的 top-k 压缩 KV 逻辑索引，经因果边界过滤后 gather。
4. **`sinks`**：可学习 sink logit，作为恒定 logit 进入 **softmax 分母**（无对应 value，详见 §3.5）。

### 3.2 核间分工与流水线

| 阶段 | AIC（Cube） | AIV（Vector） |
|---|---|---|
| Vec0 | — | 按 block table 解 PageAttention，从 GM 读取 packed KV，FP8 反量化，写入 L1/GM |
| Bmm1 | `Q @ K̃^T` | — |
| Vec1 | — | Flash Softmax：`scale`、`max`、`exp`、`sum`（含 sink 初始化） |
| Bmm2 | `P @ Ṽ` | — |
| Vec2 | — | 多轮 FlashUpdate / LastDiv，写回 GM |

`metadata`（shape `[1024]`）由 AICPU 元算子预计算，存放 FA（Cube）与 FD（Vector）的分核信息（`FA_METADATA_SIZE=9`、`FD_METADATA_SIZE=8`，共 `36 AIC + 72 AIV` 核）。PyTorch 参考实现不模拟分核，直接按 batch/position 计算。

### 3.3 KV 反量化流程

对应 kernel `CastScale` + `AntiquantVFFp8D448` + rope 拼接：

```text
kv_rope   = packed[0:128]    # 64 个 bf16
kv_nope   = packed[128:576]  # 448 个 fp8_e4m3
scales    = packed[576:583]  # 7 个 fp8_e8m0

for tile in 0..6:
    nope_dequant[tile*64:(tile+1)*64] = bf16(fp8_e4m3(nope[tile*64:(tile+1)*64]) * fp32(scale[tile]))

k = concat([nope_dequant, kv_rope])   # [512]，nope 在前、rope 在后
v = k                                  # shared KV
```

### 3.4 Query ↔ KV 位置对齐（关键）

mask 规则不是直接基于 query 在 batch 内的位置 `i`，而是基于它在 **KV 轴上的锚点**。来自 kernel `GetSingleCoreParam`：

```text
nextTokensPerBatch = actualS2Size - actualS1Size          # = S2 - S1
preTokensPerBatch  = oriWinLeft - nextTokensPerBatch       # （clip 到 [0, S1]）
```

随后 `ComputeS2LoopInfo` 计算每个 query token（batch 内行偏移 `cubeSOuterOffset = i`）的 KV 行范围：

```text
s2LineStartIdx = i + nextTokensPerBatch - oriWinLeft
s2LineEndIdx   = i + nextTokensPerBatch + 1            # （s1RealSize = 1 时）
```

即 **query token `i` 锚定在 KV 位置 `i + nextTokensPerBatch`**。当 `S2 == S1`（无前缀 prefill）时 `nextTokensPerBatch = 0`，退化为常见的 `[i-127, i]` 窗口；但当 `S2 > S1`（带前缀的 prefill，或 decode 中 `S1=1, S2=seq_len`）时，窗口整体右移 `nextTokensPerBatch`。

> ⚠️ 这是参考实现的关键点。decode 场景（`S1=1, S2=L`）下唯一的 query token 应 attend KV `[L-1-127, L-1]`（最近 128 个），而非位置 0。初版实现用 `q_pos = arange(T)` 直接与 `kv_pos = arange(S2)` 对齐，在 `S2 != S1` 时窗口完全错位，已修正为 `q_kv_pos = q_pos + (S2 - S1)`。

### 3.5 Mask 规则

记 `q_kv_pos[i] = i + nextTokensPerBatch` 为 query token `i` 的 KV 轴锚点。

#### ori_kv —— Band / Sliding Window（`ori_mask_mode=4`）

```text
q_kv_pos[i] - 127 <= j <= q_kv_pos[i] + 0      （oriWinLeft=127, oriWinRight=0）
```

#### cmp_kv —— Right-down Causal（`cmp_mask_mode=3`，`cmp_ratio=128`）

kernel 以 **exclusive 上界** `s2CmpLineEndIdx = s2LineEndIdx // cmpRatio` 划定 cmp 窗口，其中 `s2LineEndIdx = q_kv_pos[i] + 1`（`ComputeS2LoopInfo` 中 `+s1RealSize` 项，单 query 行 `s1RealSize=1`）。cmp 位置 `j` 参与 iff：

```text
j < (q_kv_pos[i] + 1) // 128      即 j <= (q_kv_pos[i] + 1) // 128 - 1
```

#### cmp_sparse_indices —— Sparse + 因果边界（`cmp_ratio=4`）

对应 kernel `GetRealCmpS2Idx` 与 `CopyInKvSparse`：

- `cmp_sparse_indices` 存的是**压缩 KV 的逻辑 token 索引 `s2Idx`**（在压缩 KV 序列内），不是物理 block id。每个 query token `i` 对应 `sparseBlockCount=K`（=512）个索引：`topkBS1Idx = (cuSeqQPrefixSum + s1oIdx) * K`。
- 物理寻址：`blockId = cmp_block_table[b, s2Idx // blockSize]`，块内偏移 `s2Idx % blockSize`（见 `GetkeyOffset`，`isCmp=true` 时用 `cmpKvStride`）。
- 因果边界（`CopyInKvSparse` 的 `s2IdLimit = (s2RealSize - actualS1Size + s1oIdx + 1) / cmpRatio`）：与 CFA 同源的 exclusive 上界，cmp 索引 `j` 有效 iff `j < (q_kv_pos[i] + 1) // 4`；超出上界视为无效（kernel 遇到 `-1` sentinel 即停止搬运）。indexer 已保证有效索引排在前半部分。

> ⚠️ cmp 因果是 **exclusive 上界 `(q_kv_pos + 1) // cmp_ratio`**，不是 `q_kv_pos // cmp_ratio`。二者在 `q_kv_pos+1` 跨越 `cmp_ratio` 整数边界时相差 1（例如 `cmp_ratio=4`、prefill `S2==S1` 下：`i=0,1,2` 不 attend 任何 cmp；`i∈[3,6]` attend cmp 0；`i>=7` attend cmp 0,1）。初版用 `q_kv_pos // cmp_ratio` 是 off-by-one，已修正并加 Case D 对拍验证。

#### seqused_kv 限制

所有 `ori_kv` 位置不能超过 `seqused_kv[b]` 指定的有效长度（gather 范围即 `[0, seqused_kv[b])`）。

### 3.6 计算精度

* `softmax_scale` 以 **bf16** 精度乘到 `Q@K^T` 结果上（kernel `ProcessVec1Vf` 的 `static_cast<T>(softmaxScale)`，`T=bf16`）。参考实现为了数值稳定在 fp32 上做 softmax，这是"参考实现用更高精度"的可接受偏差，与 kernel 的 bf16 中间结果会有 bf16 量级误差。
* `Q@K^T`（Bmm1）与 `P@V`（Bmm2）在 Cube 上以 bf16/fp32 累加（kernel L0C），参考实现用 `torch.matmul`（bf16 输入、内部累加精度由 PyTorch 决定）。

### 3.7 Learnable Sink 语义（重点）

`sinks` 是 shape `[N1]` 的 float32 **可学习参数**（`deepseek_v4.py`：`self.attn_sink = nn.Parameter(torch.empty(n_heads, dtype=torch.float32))`），**不是 KV token**。

在 kernel `ProcessVec1` 中，当 `s2LoopCount == 0 && isSinks` 时，Flash softmax 的 running 统计量初始化为：

```text
maxUb  ← sinks[h]          # running max 初值 = sink logit（Flash m0）
sumUb  ← 1.0               # = exp(sinks - max) = exp(0) = 1
```

后续每个真实 KV 块按标准 Flash online-softmax 更新 `max/sum/exp`：`m = max(m, block_max)`、`s = s_old * exp(m_old - m) + Σ exp(score - m)`。Bmm2 只对真实 KV 块累加 `P @ V`。最终 `out = accum_v / sumUb`，而 `sumUb` 含有初始的 `exp(sink - m_final)` 项。

因此 sink **只进入 softmax 分母，不进入分子（无对应 value）**，且 running max 是 `max(所有真实 score, sink)` 的联合最大值（参考实现据此实现，见 `_flash_attention_with_sink`）：

```text
m   = max( sink[h], max_j score_ij )
Z   = exp(sink[h] - m) + Σ_j exp(score_ij - m)
O_i = Σ_j ( exp(score_ij - m) / Z ) · v_j
```

> ⚠️ 初版实现曾把 `sinks` 当作 `[num_sinks, 512]` 的 KV token 拼接进 K/V，且 rowmax 用了 `clamp_min(0)`，两者都已修正：sink 改为进分母的虚拟 logit，rowmax 与 sink 联合取 max（不 clamp，否则 sink 主导时数值会偏）。

---

## 4. PyTorch 参考实现

参考实现位于同目录下：

```text
kv_quant_sparse_attn_sharedkv_reference.py
```

实现要点（与 kernel 数值语义对齐）：

* `dequant_packed_kv`：还原 640B packed KV 的 per-tile（64-dim）FP8 反量化，输出 `[nope(448) | rope(64)]`。
* `gather_page_attention_kv`：还原 PageAttention 的 block-table gather（`s2Idx // blockSize` 查表 + `s2Idx % blockSize` 块内偏移）。
* `build_band_mask` / `build_right_down_causal_mask`：基于 **KV 轴锚点 `q_kv_pos = q_pos + (S2 - S1)`** 构造 band 与 right-down causal mask（见 §3.4）。
* `_flash_attention_with_sink`：带 learnable sink 的 Flash attention；rowmax 与 sink **联合取 max**（不 clamp），sink 作为虚拟 logit 进 softmax 分母（见 §3.6）。
* `kv_quant_sparse_attn_sharedkv_pytorch`：主入口，支持 `cmp_ratio=1/4/128` 三种模式。
  - `cmp_ratio=4`：逐 query token 用 `cmp_sparse_indices` gather 各自 top-k 压缩 KV，按 exclusive 因果边界 `j < (q_kv_pos[i]+1)//4` 过滤（见 §3.5）。
  - `cmp_ratio=128`：拼接全量压缩 KV 尾部，应用 right-down causal mask。
  - `cmp_ratio=1`：纯 SWA。
* `pack_random_kv`（自测辅助）：生成**合法**的 640B packed KV（随机 uint8 直接 reinterpret 会命中 fp8_e4m3fn 的 `0xFF` NaN 槽位），通过真实 bf16→fp8 量化再打包。

> 仅保证数值语义一致，不做性能优化；不能用于生产性能评估。

直接运行可执行 CPU 自测，覆盖四种场景：
* **Case A**：prefill（`S2 == S1`，`nextTokensPerBatch = 0`），三种 `cmp_ratio`。
* **Case B**：decode（`S1=1, S2=300`）—— 验证 query↔KV 对齐：输出与"显式取最后 128 个 ori_kv 窗口"的独立 SWA 实现对拍（bf16 级误差）。
* **Case C**：sink 语义对拍 —— Flash 实现 vs 把 sink 当虚拟 KV token（score=sink、value=0）的朴素 softmax 实现。
* **Case D**：cmp 因果边界对拍 —— `cmp_ratio=4`，验证 exclusive 上界 `(q_kv_pos+1)//4`：`i=0,1,2` 不 attend cmp；`i∈[3,6]` attend cmp 0；`i>=7` attend cmp 0,1，与独立实现逐 query 对拍。

```bash
python kv_quant_sparse_attn_sharedkv_reference.py
```

主要函数签名（**与 NPU 算子 `csrc/torch_binding.cpp` 的 `npu_kv_quant_sparse_attn_sharedkv` schema 参数名、顺序、默认值完全一致**，可按位置或关键字调用，与 `dsa_v1.py` 调用 `attn_op(...)` 的方式一一对应）：

```python
def kv_quant_sparse_attn_sharedkv_pytorch(
    q: torch.Tensor,                                  # [T,N1,512](TND) / [B,S1,N1,512](BSND), bf16
    kv_quant_mode: int = 1,                           # op-fixed 1 (per-tile fp8_e4m3)
    ori_kv: Optional[torch.Tensor] = None,            # [num_blocks,block_size,1,640], uint8(fp8 packed)
    cmp_kv: Optional[torch.Tensor] = None,            # same layout/dtype as ori_kv
    ori_sparse_indices: Optional[torch.Tensor] = None,# reserved, unused
    cmp_sparse_indices: Optional[torch.Tensor] = None,# [T,1,K], int32 (cmp_ratio=4)
    ori_block_table: Optional[torch.Tensor] = None,   # [B,max_blocks], int32
    cmp_block_table: Optional[torch.Tensor] = None,   # [B,max_blocks], int32
    cu_seqlens_q: Optional[torch.Tensor] = None,      # [B+1], int32 (TND)
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None, # reserved, unused
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None, # reserved, unused
    seqused_q: Optional[torch.Tensor] = None,         # reserved, unused
    seqused_kv: Optional[torch.Tensor] = None,        # [B], int32
    sinks: Optional[torch.Tensor] = None,             # [N1], float32 learnable sink logit
    metadata: Optional[torch.Tensor] = None,          # [1024], int32 (AICPU 分核, 不模拟)
    tile_size: int = 0,                               # op-fixed 64 (0->64)
    rope_head_dim: int = 0,                           # op-fixed 64 (0->64)
    softmax_scale: float = 0.0,                       # attention scale
    cmp_ratio: int = 0,                               # 1(SWA) / 4(sparse) / 128(dense)
    ori_mask_mode: int = 4,                           # op-fixed 4 (band/SWA)
    cmp_mask_mode: int = 3,                           # op-fixed 3 (right-down causal)
    ori_win_left: int = 127,                          # op-fixed 127
    ori_win_right: int = 0,                           # op-fixed 0
    layout_q: str = "BSND",                           # TND / BSND
    layout_kv: str = "PA_ND",                         # op-fixed PA_ND
    return_softmax_lse: bool = False,                 # 是否返回 softmax_lse
) -> tuple[torch.Tensor, torch.Tensor]:
    # 返回 (attention_out, softmax_lse)，与算子 2-tuple 输出一致。
    # return_softmax_lse=False 时 softmax_lse 为空 tensor (shape [0])，与算子 InferShape 一致。
```

> 对齐说明：参数顺序与默认值逐字取自 `torch_binding.cpp` 第 2766–2793 行的 `ops.def(...)`。`dsa_v1.py` 的四处调用（2111/2136/2431/2453 行）使用关键字传参，本参考实现完全兼容其调用方式。`ori_sparse_indices`/`cu_seqlens_ori_kv`/`cu_seqlens_cmp_kv`/`seqused_q`/`metadata` 在算子 README 中标注为"预留/分核"，参考实现接收但内部不使用。op-fixed 属性（`kv_quant_mode=1`、`tile_size=64`、`rope_head_dim=64`、`ori_mask_mode=4`、`cmp_mask_mode=3`、`ori_win_left=127`、`ori_win_right=0`、`layout_kv=PA_ND`）若传入非法值会在入口校验阶段 `ValueError`，对齐 host check 行为。

### 4.1 `kv_quant_sparse_attn_sharedkv_pytorch` 主函数逐步解析

主函数按"校验 → 归一化布局 → 逐 batch 处理 → 组装输出"四段执行。下方逐步对照 kernel 行为说明。

#### 步骤 0：入参校验（对齐 host check）

入口处一连串 `ValueError` 校验，复刻 `kv_quant_sparse_attn_sharedkv_check_*.cpp` 的 op-fixed 约束：

| 校验项 | 允许值 | 对应 host check |
|---|---|---|
| `kv_quant_mode` | `1` | `CheckFeatureAntiquantAttr`（per-tile fp8_e4m3） |
| `tile_size` | `0`(默认)/`64` | `CheckFeatureAntiquantAttr` |
| `rope_head_dim` | `0`(默认)/`64` | `CheckFeatureAntiquantAttr` |
| `ori_mask_mode` / `cmp_mask_mode` | `4` / `3` | `CheckSingleParaSparseMode` |
| `ori_win_left` / `ori_win_right` | `127` / `0` | `CheckFeatureWinKV` |
| `layout_q` / `layout_kv` | `TND`/`BSND` / `PA_ND` | `CheckFeatureAntiquantLayout` |
| `q.shape[-1]` | `512` | `CheckFeatureAntiquantShape` |
| `ori_kv.shape[-1]` | `640` | `CheckSingleParaKey`（`dSizeOriKvInput_ != 640`） |
| `cmp_ratio` | `1`/`4`/`128` | README 约束 |
| `cmp_ratio=4` 依赖 | `cmp_kv`+`cmp_block_table`+`cmp_sparse_indices` | 场景三必备入参 |
| `cmp_ratio=128` 依赖 | `cmp_kv`+`cmp_block_table` | 场景二必备入参 |
| `ori_block_table`/`seqused_kv` | 必传 | PageAttention + KV 长度必备 |
| TND 下 `cu_seqlens_q` | 必传 | `CheckSingleParaCuSeqLensQ` |

> 预留参数 `ori_sparse_indices`/`cu_seqlens_ori_kv`/`cu_seqlens_cmp_kv`/`seqused_q`/`metadata` 不做语义校验，仅接收以保持签名一致（README 标注"预留/分核"）。

#### 步骤 1：BSND 布局归一化为 TND

核心计算逻辑按 TND（`[T, N1, D]`）编写。若 `layout_q="BSND"`，先把 `[B, S1, N1, D]` flatten 成 `[B*S1, N1, D]`，并构造对应的 `cu_seqlens_q = [0, S1, 2*S1, ..., B*S1]`，使后续循环统一按 TND 处理。`bsnd` 标志位记录原始布局，供步骤 4 还原输出 shape。

> 对应 kernel：BSND/TND 两种 `layout_q` 由 tiling key 区分，`GetSingleCoreParam` 中 `actualS1Size` 在 TND 下取 `cuSeqlensQAddr[sIdx+1]-cuSeqlensQAddr[sIdx]`，BSND 下取 `constInfo.s1Size`。参考实现用 flatten 把两者统一到 TND 路径。

#### 步骤 2：逐 batch 循环（`for b in range(B)`）

对每个 batch `b`：

1. **切出该 batch 的 query**：`q_b = q[cu_seqlens_q[b]:cu_seqlens_q[b+1]]`，query 数 `T_b`，query 位置 `q_pos_b = arange(T_b)`。
2. **取该 batch 的 ori_kv 有效长度**：`seq_len = seqused_kv[b]`（对应 kernel `actualS2Size = actualSeqKvlenAddr[sIdx]`）。

#### 步骤 3：query ↔ KV 位置对齐（关键）

```python
next_tokens = seq_len - T_b            # = S2 - S1 = kernel nextTokensPerBatch
q_kv_pos   = q_pos_b + next_tokens     # query 在 KV 轴的锚点
```

这是 §3.4 的核心：query token `i` 锚定在 KV 位置 `i + (S2-S1)`。`S2==S1`（无前缀 prefill）时 `next_tokens=0` 退化为 `[i-127, i]`；`S2>S1`（decode 或带前缀）时窗口整体右移。

#### 步骤 4：ori_kv 的 gather + 反量化（共享 KV）

```python
ori_positions  = arange(seq_len)                              # [0, S2)
ori_kv_packed  = gather_page_attention_kv(ori_kv, ori_block_table[b:b+1], ori_positions).squeeze(1)
ori_kv_dequant = dequant_packed_kv(ori_kv_packed)             # [S, 512]
k_ori = v_ori = ori_kv_dequant                                # shared KV (K̃ = Ṽ)
```

* `gather_page_attention_kv`：按 `s2Idx → block_table[b, s2Idx//blockSize]` + `s2Idx%blockSize` 物理寻址（对应 kernel `GetkeyOffset` 的 PA 路径）。
* `dequant_packed_kv`：640B 打包 → `[nope(448) | rope(64)]` 的 bf16（对应 kernel `DequantKv`，§3.3）。
* **shared KV**：同一份反量化结果同时充当 K 和 V（MLA 类模型特性）。

同时预计算 band mask（§3.5）：`mask_ori = build_band_mask(q_kv_pos, ori_positions, 127, 0)`，query `i` attend ori_kv `j ∈ [q_kv_pos[i]-127, q_kv_pos[i]]`。

#### 步骤 5：按 `cmp_ratio` 分支处理压缩 KV

**分支 A —— `cmp_ratio == 4`（SCFA，稀疏）**：每个 query token 拥有独立的 top-k 压缩 KV 索引，**逐 query 单独计算**（因为各 query 的 KV 集合不同）。

```python
cmp_pos_limit = (q_kv_pos + 1) // cmp_ratio - 1     # exclusive 因果上界 (§3.5)
indices_b     = cmp_sparse_indices[q_start:q_end, 0, :]   # [T_b, K]
for t in range(T_b):
    valid   = (indices_b[t] >= 0) & (indices_b[t] <= cmp_pos_limit[t])  # 因果过滤
    # gather + dequant 该 query 的 top-k cmp KV，与 SWA 窗口内的 ori_kv 拼接
    k_t = cat([k_ori[~mask_ori[t]], k_cmp[valid]], dim=0)
    out_t = _attend_single_query(q_b[t:t+1], k_t, v_t, softmax_scale, sink=sinks)
```

要点：
* 因果边界是 **exclusive `(q_kv_pos+1)//4`**（§3.5），不是 `q_kv_pos//4`（off-by-one，见 Case D）。
* `cmp_sparse_indices` 存的是 cmp_kv 逻辑 token 索引，经 `cmp_block_table` 二次寻址 gather（§3.5）。
* 逐 query 循环是因为各 query 的 cmp KV 集合不同（对应 kernel `GetRealCmpS2Idx` 用 `s1oIdx` 索引 `topkBS1Idx`）。

**分支 B —— `cmp_ratio ∈ {1, 128}`（SWA / CFA）**：所有 query token **共享同一份 KV**（ori_kv，可选拼接全量 cmp_kv），可一次性矩阵化计算。

```python
k_parts = [k_ori]; v_parts = [v_ori]
if cmp_ratio == 128 and seq_len // 128 > 0:
    cmp_kv_dequant = dequant_packed_kv(gather(cmp_kv, cmp_block_table[b], arange(cmp_len)))
    k_parts.append(cmp_kv_dequant); v_parts.append(cmp_kv_dequant)   # 尾部拼接全量压缩 KV

k_all = cat(k_parts); v_all = cat(v_parts)         # [S (+cmp_len), 512]
mask  = build_band_mask(q_kv_pos, kv_pos_all, 127, 0)
if cmp_ratio == 128:
    mask[:, -cmp_len:] = build_right_down_causal_mask(q_kv_pos, cmp_positions, 128)  # 尾部覆盖为 right-down causal

out_b = _flash_attention_with_sink(q_b, k_all, v_all, softmax_scale, mask, sinks)
```

要点：
* `cmp_ratio=1`：纯 SWA，不拼接 cmp_kv。
* `cmp_ratio=128`：尾部拼接 `[0, S2/128)` 全量压缩 KV，对该尾部应用 right-down causal mask（`j < (q_kv_pos+1)//128`），ori_kv 部分仍是 band mask。
* mask 在 ori/cmp 拼接张量上构造：前 `S` 列 band，后 `cmp_len` 列 right-down causal。

#### 步骤 6：单次 attention 计算（`_flash_attention_with_sink`）

无论哪个分支，最终都归约到一次"带 learnable sink 的 Flash attention"（§3.7）：

```text
scores = Q @ K^T * softmax_scale                  # [T, N, S]
scores = masked_fill(scores, mask, -inf)
m      = max(real_max, sink)                       # 与 sink 联合取 max（不 clamp）
e      = exp(scores - m)
Z      = e.sum(-1) + exp(sink - m)                 # sink 进分母，无对应 value
attn   = e / Z
out    = attn @ V
```

对应 kernel 的 `Bmm1(Q@K^T)` → `Vec1(Flash Softmax)` → `Bmm2(P@V)` 三段（§3.2）。

#### 步骤 7：组装输出与 softmax_lse

```python
attn_out = cat(outputs, dim=0)                    # [T, N1, 512]
if bsnd: attn_out = attn_out.reshape(B_in, S1_in, N_q, 512)   # 还原 BSND
softmax_lse = zeros((0,)) if not return_softmax_lse else zeros((N_q, T))
return attn_out, softmax_lse
```

* BSND 输出还原为 `[B, S1, N1, 512]`。
* 返回值是与算子一致的 **2-tuple `(attention_out, softmax_lse)`**。`return_softmax_lse=False` 时 `softmax_lse` 为空（shape `[0]`，对应算子 `InferShapeKvQuantSparseAttnSharedkv` 中 `returnSoftmaxLse=false` 分支）；True 时返回占位 lse（**非数值验证**，算子当前也未真正支持该输出）。

#### 主函数控制流总览

```text
入参校验 (步骤0)
    │
    ▼
BSND → TND 归一化 (步骤1)
    │
    ▼
for b in range(B): ──────────────────────────────────────────┐
  ├─ 切 query / 取 seq_len (步骤2)                            │
  ├─ q_kv_pos = q_pos + (S2-S1) (步骤3, §3.4)                │
  ├─ ori_kv gather+dequant, shared KV (步骤4)                │
  ├─ cmp 分支 (步骤5):                                        │
  │   ├─ cmp_ratio=4 : 逐 query, exclusive 因果过滤 (SCFA)   │
  │   └─ cmp_ratio∈{1,128}: 共享 KV, 矩阵化 (SWA/CFA)        │
  ├─ _flash_attention_with_sink (步骤6, §3.7)                │
  └─ 收集 out_b                                              │
    │ ◀─────────────────────────────────────────────────────┘
    ▼
cat + BSND 还原 + softmax_lse 占位 (步骤7)
    │
    ▼
return (attention_out, softmax_lse)
```

---

## 5. 与 `dsa_v1.py` 的调用关系

算子在 `vllm_ascend/attention/dsa_v1.py` 中通过 `DeviceOperator.get_dsa_sparse_attn_op()` 获取（实际为 `torch.ops._C_ascend.kv_quant_sparse_attn_sharedkv`）。`self.attn_sink`（shape `[n_heads]`，float32）作为 `sinks` 传入。

调用位置：

| 行号 | 路径 | `cmp_ratio` |
|---|---|---|
| 2111 | `_forward_prefill` | `4`（SCFA，传 `cmp_sparse_indices`） |
| 2136 | `_forward_prefill` | `128`（CFA） |
| 2413 | `_forward_decode` | `≤1`（SWA-only） |
| 2431 | `_forward_decode` | `4`（SCFA） |
| 2453 | `_forward_decode` | `128`（CFA） |

典型调用（以 prefill `cmp_ratio=4` 为例）：

```python
attn_output = attn_op(
    q,                                        # [T, N_q, 512], bf16
    ori_kv=swa_kv_cache,                      # PA_ND, fp8_e4m3fn packed
    cmp_kv=compress_kv_cache,                 # PA_ND, fp8_e4m3fn packed
    cmp_sparse_indices=compress_topk_idxs,    # [T, 1, 512], int32
    ori_block_table=swa_prefill_metadata.block_table,
    cmp_block_table=compressor_prefill_metadata.block_table,
    cu_seqlens_q=actual_seq_lengths_query,
    seqused_kv=actual_seq_lengths_key,
    sinks=self.attn_sink,                     # [N_q], float32 learnable sink
    metadata=common_prefill_metadata.sas_metadata,
    softmax_scale=self.softmax_scale,
    cmp_ratio=self.compress_ratio,            # 4
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=self.window_size - 1,        # 127
    ori_win_right=0,
    layout_q="TND",
    layout_kv="PA_ND",
    **extra_attn_kwargs,                      # 含 kv_quant_mode=1, tile_size=64,
                                             #      rope_head_dim=64，可能含 cu_seqlens_cmp_kv
)[0]                                         # 返回 (attention_out, softmax_lse)，取第一个
```

> 注意 `cu_seqlens_q` 在 `dsa_v1.py` 中传入的是 `actual_seq_lengths_query`（每 batch 的 q 长度数组），与算子 README 描述的"累积和 `[B+1]`"在 TND 路径下由 kernel 内部按 `cuSeqlensQAddr` 处理；参考实现按"累积和"语义实现，调用方需保证语义一致。

---

## 6. 注意事项

1. **反量化是强制的，KV 不支持 bf16**：host check 限定 `ori_kv`/`cmp_kv` 只能是 INT8 或 FLOAT8_E4M3FN、D 维必须 640、`kv_quant_mode=1`、`tile_size=64`。kernel 无条件执行 fp8→bf16 反量化（见 §2.2）。Python 侧用 `uint8` 视图承载该 buffer，与算子的单字节存储数值等价。`0xFF` 字节在 fp8_e4m3fn 中是 NaN，因此测试数据需用合法量化值构造（见 `pack_random_kv`），不能用裸 `randint(0,255)`。
2. **Shared KV**：`K̃ = Ṽ`，反量化后的 `[nope | rope]` 张量同时作为 K 与 V（MLA 类模型）。
3. **Sink 语义**：`sinks` 是 `[N1]` 的 learnable sink logit，**只进 softmax 分母**，非 KV token；rowmax 与 sink 联合取 max。
4. **Query↔KV 对齐**：mask 基于 KV 轴锚点 `q_kv_pos = q_pos + (S2 - S1)`，而非 query 在 batch 内的位置。`S2 != S1`（decode、带前缀 prefill）时窗口整体右移 `nextTokensPerBatch`。
5. **cmp_sparse_indices**：存的是压缩 KV 的**逻辑 token 索引**（非物理 block id），需通过 `cmp_block_table` 二次寻址；逐 query 独立，受 exclusive 因果上界 `j < (q_kv_pos[i] + 1) // cmp_ratio` 约束（见 §3.5）。
6. **PageAttention**：参考实现按 batch 单独 gather；真实 kernel 在 AIV 上通过 block table 动态寻址并跨 block 连续搬运。
7. **Metadata**：`metadata` 是 AICPU 分核结果，参考实现未模拟分核，直接按 batch/position 计算。
8. **性能**：参考实现仅用于算法验证与数值对拍，不能用于生产性能评估。

---

## 7. 未来可扩展

* 接入真实 `torch.ops._C_ascend.kv_quant_sparse_attn_sharedkv` 做端到端数值对拍（含 sink、三种 cmp_ratio、decode/前缀场景）。
* 增加更大 batch、更长序列、不同 `index_topk` 的单元测试。
* 将反量化逻辑替换为真实 `npu_quant_dequant` 以验证 FP8 round-trip 误差。
* 对齐 `cu_seqlens_q` 在 `dsa_v1.py` 实际入参（每 batch 长度 vs 累积和）的语义。
