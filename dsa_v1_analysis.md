# `vllm_ascend/attention/dsa_v1.py` 详细解析

## 1. 文件总览

`dsa_v1.py` 实现了 **DeepSeek-V4 的 Dynamic Sparse Attention（DSA）** 在昇腾 NPU 上的 vLLM v1 attention 后端。它继承自 `DSAAttentionImpl`（`vllm_ascend/attention/abstract.py`），主要包括：

- `AscendDSABackend`：vLLM v1 attention backend 注册类。
- `AscendDSAMetadataBuilder`：为 prefill / decode / drafting / graph capture 构建 attention metadata。
- `AscendDSAImpl`：DSA attention 的前向计算实现，包括：
  - MLA prolog（q/kv 投影、RMSNorm、rope、KV scatter）。
  - 压缩 KV 的 compressor 与 indexer（top-k 稀疏索引）。
  - 稀疏 attention 计算。
  - o-proj 输出投影。

---

## 2. 模块级辅助函数

### 2.1 `dsv4_dsa_overlap_stream()`

```python
def dsv4_dsa_overlap_stream() -> torch.npu.Stream:
```

- 懒加载一个全局辅助 NPU stream `_DSV4_DSA_OVERLAP_STREAM`。
- 用于 DSA 多流并行（main stream + aux stream），让 Vector/Cube/AIV 引擎并发工作。

---

### 2.2 Hadamard 变换相关函数

#### `hadamard_transform_ref(x, hadamard, scale=1.0)`

- 对输入 `x` 做参考实现的 Hadamard 变换。
- 流程：
  1. 取最后一维 `dim`，向上补齐到最近的 2 的幂 `dim_padded`。
  2. 用 `F.linear` 与 Hadamard 矩阵相乘。
  3. 乘以 `scale` 并截回原始维度。

#### `rotate_activation(x, hadamard)`

- 对激活值做旋转：调用 `hadamard_transform_ref`，scale 为 `hidden_size ** -0.5`。
- 用于 compressor / indexer 的 KV 旋转。

#### `hadamard_linear(x, hadamard)`

- Hadamard 变换的第一阶段：只做 `F.linear`。
- 返回 `(linear_output, original_shape, original_dim)`，便于后续在另一个 stream 做 scale/reshape。

#### `hadamard_scale(out, x_shape, dim, scale=1.0)`

- Hadamard 变换的第二阶段：scale 并 reshape 回原始形状。
- 设计目的：第一阶段可在 main stream 与 aux stream 的 `kv_scatter` 并行；第二阶段在 aux stream 完成后执行。

---

### 2.3 `_is_w8a8_dynamic(linear)`

- 判断一个线性层是否使用了 `AscendW8A8DynamicLinearMethod` 动态量化。
- 用于在 q/kv 投影路径中选择 W8A8 量化 matmul 还是普通 matmul。

---

### 2.4 `pad_to_blocks(x, length_list, block_size=128)`

- 将变长 packed tensor `x`（shape `[sum(length_list), n, d]`）按块补齐。
- 输出 shape `[total_blocks, block_size, n, d]`，每个请求的数据按 `block_size` 分块，不足补零。
- 用于 prefill 阶段将 ragged tensor 转为固定块大小，便于后续 NPU 算子处理。

---

## 3. `AscendDSABackend`

vLLM v1 attention backend 注册类。

### 3.1 `get_name()`

- 返回 `"ASCEND_DSA"`（v1 runner）或 `"FLASH_ATTN"`（v2 runner，为了绕过 vLLM v2 的断言）。

### 3.2 `get_builder_cls()`

- 如果启用了 DSA context parallel（`enable_dsa_cp()`），返回 `AscendDSACPMetadataBuilder`。
- 否则返回 `AscendDSAMetadataBuilder`。

### 3.3 `get_impl_cls()`

- 如果启用了 DSA context parallel，返回 `AscendDSACPImpl`。
- 否则返回 `AscendDSAImpl`。

### 3.4 `get_kv_cache_shape(...)`

- 返回 KV cache 形状：`(num_blocks, block_size, num_kv_heads, head_size)`。

### 3.5 `get_scale_shape(...)`

- 返回 scale cache 形状：`(num_blocks, block_size, scale_size)`。

### 3.6 `get_supported_kernel_block_sizes()`

- 返回支持的 KV cache block size 列表：`[2, 4, 8, 16, 32, 64, 128]`。

---

## 4. 元数据 dataclass

### 4.1 `AscendDSAPrefillMetadata`

Prefill 阶段专用元数据：

| 字段 | 说明 |
|---|---|
| `attn_mask` | 注意力 mask（DSA 中通常为 None） |
| `query_lens` | 每个请求的 query token 数 |
| `seq_lens` | 每个请求的总序列长度 |
| `context_lens` | 上下文长度 |
| `input_positions` | 输入位置编码 |
| `query_start_loc` | query 的累积长度 |
| `block_table` | KV cache 块表 |
| `slot_mapping` | KV cache slot 映射 |
| `block_size` | KV cache 块大小 |
| `max_query_len` / `max_seq_lens` | 最大 query / 序列长度 |
| `num_compressed_tokens` | 压缩 token 数量 |
| `sin` / `cos` | RoPE 正弦/余弦 |
| `full_compress_sin` / `full_compress_cos` | 完整压缩 RoPE |
| `start_pos` | 每个请求在序列中的起始位置 |
| `num_reqs_actual` | 实际请求数 |
| `sas_metadata` | 稀疏 attention 元数据 |
| `qli_metadata` | Quant Lightning Indexer 元数据 |
| `cu_c4_cmp_seqlen_list` / `cu_c128_cmp_seqlen_list` | C4/C128 压缩序列长度累积 |

### 4.2 `AscendDSADecodeMetadata`

Decode 阶段专用元数据，字段与 prefill 类似，额外包含：

| 字段 | 说明 |
|---|---|
| `query_start_loc_cpu` | CPU 上的 query start loc |
| `seq_lens_list` | decode 请求序列长度列表 |
| `cp_seq_len` | context parallel 序列长度 |
| `batch_seq_mask` | batch 序列 mask |

### 4.3 `AscendDSAMetadata`

顶层 metadata，聚合 prefill 与 decode：

| 字段 | 说明 |
|---|---|
| `num_actual_tokens` | 实际 token 数（不含 padding） |
| `slot_mapping` / `query_start_loc` / `seq_lens` / `block_tables` | 通用信息 |
| `sin` / `cos` | RoPE（按 layer 索引的 dict） |
| `num_decodes` / `num_decode_tokens` / `num_prefills` | decode/prefill 数量 |
| `attn_state` | `AscendAttentionState`（ChunkedPrefill / DecodeOnly / SpecDecoding） |
| `decode` / `prefill` | 子 metadata |
| `hadamard` | Hadamard 矩阵 |
| `start_pos` | 起始位置 |

---

## 5. `AscendDSAMetadataBuilder`

继承自 `AttentionMetadataBuilder[AscendDSAMetadata]`，负责为每次 forward 构建 DSA 需要的 metadata。

### 5.1 `__init__(...)`

- 保存 `kv_cache_spec`、`vllm_config`、`model_config`、`device`。
- 计算 `max_blocks`、`rope_dim`。
- 根据是否 A5 决定 `slot_mapping` 形状：
  - A5: `(max_num_batched_tokens,)`
  - 其他: `(max_num_batched_tokens, 2)`
- 处理 speculative decoding 的 `spec_slot_mapping` 和 `spec_sas_metadata`。
- 初始化 `start_pos_prefill`、`start_pos_decode`、`decode_sas_metadata`、`decode_qli_metadata` 等 buffer。
- 如果模型类型是 `deepseek_v4`，生成全局 Hadamard 矩阵（缓存在类变量中，避免重复创建）。
- 初始化 `slot_mapping`、`cu_seqlens_ori_kv`、`cu_seqlens_cmp_kv` 等空 tensor。

### 5.2 `reorder_batch(input_batch, scheduler_output)`

- 将 batch 重排：decode 请求放前面，prefill 请求放后面。
- 依据 `num_tokens <= decode_threshold` 判断请求是 decode 还是 prefill。
- 通过 `input_batch.swap_states` 做最小化交换。
- 返回是否修改了 batch。

### 5.3 `set_num_actual_tokens(...)`

- 保存 `common_attn_metadata.num_actual_tokens`。

### 5.4 `_num_compressor_metadata_rows(build_step, common_attn_metadata)`

- 计算压缩元数据的行数：
  - prefill: `min(num_prefill_tokens, num_prefill_tokens // ratio + num_prefills)`
  - decode: `min(num_decode_tokens, num_decode_tokens // ratio + num_decodes)`

### 5.5 `build(common_prefix_len, common_attn_metadata, fast_build=False, **kwargs)`

主构建入口：

1. 获取 `num_reqs`、`query_start_loc`、`num_reqs_actual`。
2. 从 kwargs 获取 `prefill_ratio_to_sas_metadata`、`decode_ratio_to_sas_metadata`、`common_ratio_to_sas_metadata`、`block_size`。
3. **第一次构建**时：
   - 调用 `split_decodes_and_prefills` 分割 decode/prefill。
   - 缓存到 `common_ratio_to_sas_metadata`。
   - 生成全局 `input_positions`、`cos/sin`、`seq_lens`、`query_lens`。
4. **后续构建**时从缓存读取。
5. 格式化 `slot_mapping`，截取 `block_table`。
6. 如果有 prefill，调用 `build_prefill_metadata`。
7. 如果有 decode，调用 `build_decode_metadata`。
8. 返回 `AscendDSAMetadata`。

### 5.6 `build_prefill_metadata(...)`

构建 prefill 子 metadata：

1. 计算 prefill 请求起始位置 `reqs_start = num_decodes`，token 起始 `tokens_start = num_decode_tokens`。
2. 缓存/读取 prefill 相关的 `input_positions`、`max_query_len`、`max_seq_lens`、`query_start_loc`、`seq_lens`、`cos/sin`。
3. 计算 `start_pos_prefill`。
4. 根据 `compressor_ratio` 分支：
   - `<= 1`：SWA 层，无压缩。
   - `== 4`：C4 压缩。
   - `> 4`（如 128）：C128 压缩。
5. 调用 `DeviceOperator.get_dsa_sparse_attn_metadata_op()` 生成 `sas_metadata`。
6. 调用 `_C_ascend.npu_vllm_quant_lightning_indexer_metadata` 生成 `qli_metadata`。
7. 返回 `AscendDSAPrefillMetadata`。

### 5.7 `build_decode_metadata(...)`

构建 decode 子 metadata：

1. 缓存/读取 decode 相关的 `query_start_loc`、`input_positions`、`cos/sin`、`query_start_loc_cpu`、`max_seq_lens`、`seq_lens_list`、`max_seqlen_kv/q`。
2. 计算 `start_pos_decode`。
3. 根据 `num_reqs_actual` 对 `start_pos_decode` 和 `block_table` 做 padding 清零。
4. 调用 `DeviceOperator.get_dsa_decode_cu_seqlens_ori_kv` 计算 decode 的 `cu_seqlens_ori_kv`。
5. 类似 prefill，根据 `compressor_ratio` 生成 `sas_metadata` 和 `qli_metadata`。
6. 返回 `AscendDSADecodeMetadata`。

### 5.8 `build_for_drafting(...)`

- 用于 speculative decoding 的 draft model。
- 限制 `compressor_ratio <= 1`（仅支持 SWA 层）。
- 分别构建 prefill 和 decode metadata，使用 `spec_slot_mapping` 保存不同 draft iteration 的 slot mapping。
- 调用 `build_prefill_metadata_for_drafting` 和 `build_decode_metadata_for_drafting`。

### 5.9 `build_prefill_metadata_for_drafting(...)`

- 简化版 prefill metadata，用于 draft iteration。
- 生成 `sas_metadata`，使用 `spec_slot_mapping[draft_index - 1]`。
- 不缓存到 `ratio_to_sas_metadata`。

### 5.10 `build_decode_metadata_for_drafting(...)`

- 简化版 decode metadata，用于 draft iteration。
- 生成 `sas_metadata` 并保存到 `spec_sas_metadata[draft_index - 1]`。
- 使用 `spec_slot_mapping[draft_index - 1]`。

### 5.11 `get_block_table_size(...)`

- prefill 返回 `num_reqs`。
- decode 返回 `num_decodes`。

### 5.12 `build_for_graph_capture(...)`

- 用于 ACL graph capture。
- 仅支持 `DecodeOnly` 和 `SpecDecoding` 状态。
- 调用 `build(...)` 生成 metadata 并设置 `attn_state`。

---

## 6. `AscendDSAImpl`

DSA attention 的核心实现类。

### 6.1 `__init__(...)`

- 保存 attention 参数：`num_heads`、`head_dim`、`rope_head_dim`、`nope_head_dim`、`n_groups`、`window_size`、`compress_ratio`、`scale` 等。
- 保存 MLA 相关权重：`wq_a`、`wq_b`、`wkv`、`q_norm`、`kv_norm`、`wo_a`、`wo_b`。
- 用 `CVLinearWrapper` 包装 `wq_a`、`wkv`、`wq_b`，实现 Vector + Cube 的混合量化 matmul。
- 保存 indexer 和 compressor 子模块引用（如果存在）。
- 读取 `AscendConfig` 中的 `multistream_dsv4_dsa_overlap` 开关。
- 初始化 index cache 相关：`skip_topk`、`topk_indices_buffer`、`use_index_cache`。

### 6.2 `update_graph_params(...)`

- DSA 不需要更新 graph params，空实现。

### 6.3 `_get_indexcache_topk_indices(num_tokens, offset=0)`

- 从 `topk_indices_buffer` 读取缓存的 top-k 索引。
- 如果维度为 2，增加一维变为 `[T, 1, K]`。

### 6.4 `_update_indexcache_topk_indices(topk_indices, offset=0)`

- 将当前层计算的 top-k 索引写入 `topk_indices_buffer`，供后续 `skip_topk` 层复用。

### 6.5 `_compute_compressor_metadata(metadata)`

- 调用 `_C_ascend.compressor_metadata` 算子，生成压缩所需的 `compress_cos`、`compress_sin`、`compress_slot_mapping`。

### 6.6 `process_weights_after_loading(...)`

- DSA 的 OTP buffer 懒加载，无需 vLLM 的 `process_weights_after_loading` 分发。

### 6.7 `rope_single(x, cos, sin, inverse=False)`

- 使用 `torch_npu.npu_rotary_mul` 对 `x` 应用 RoPE。
- 支持 TND 和 BSND 两种 layout。
- `inverse=True` 时 sin 取反，用于输出后的 inverse rope。

### 6.8 `_forward_o_proj(o_proj_input, output)`

DSA 的 o-projection 输出投影，支持三种路径：

1. **A5 (Ascend950)**：
   - 使用 `npu_dynamic_mx_quant` 量化到 FP8。
   - 使用 `npu_transpose_quant_batchmatmul` 做量化 batch matmul。
   - 最后经过 `wo_b`。

2. **oproj_tp_enable()（OTP，o-proj tensor parallel）**：
   - 通过 `get_otp_group()` 获取 OTP 通信组。
   - 使用静态 buffer 做 `all_to_all_single` 和 `reduce_scatter_tensor`，保证 ACL graph replay 时地址稳定。
   - 先 `wo_a`，再 `wo_b`。

3. **olora_tp_enable() 或默认路径**：
   - 默认路径使用 `npu_transpose_batchmatmul` 计算 `wo_a`，再经过 `wo_b`。

### 6.9 `forward(layer_name, hidden_states, kv_cache, attn_metadata, need_gather_q_kv, output)`

DSA attention 主入口：

1. 如果 `attn_metadata is None`（profiling run）：
   - 若启用 OTP，用 zero input 跑一遍 `_forward_o_proj` 以 capture HCCL。
   - 否则直接 zero output。
2. 将 `hidden_states` gather/unpad。
3. 分割为 `decode_hidden_states`（前 `num_decode_tokens`）和 `prefill_hidden_states`。
4. 分配 `o_proj_input` buffer。
5. 如果有 prefill，调用 `_forward_prefill` 计算 prefill 输出，写入 `o_proj_input[decode_tokens:actual_tokens]`。
6. 如果有 decode，调用 `_forward_decode` 计算 decode 输出，写入 `o_proj_input[:decode_tokens]`。
7. 对 `o_proj_input` 做 inverse partial rotary mul（恢复 nope 部分）。
8. 调用 `_forward_o_proj` 得到最终输出。

### 6.10 `_mla_prolog_multistream(hidden_states, cos, sin, swa_kv_cache, slot_mapping, is_prefill=False)`

**多流 MLA prolog**，将 q 和 kv 的计算拆分到 main stream 和 aux stream 并行：

| 阶段 | Main Stream | Aux Stream |
|---|---|---|
| Part1 | q_quant[V] → q_a_down[C] | kv_quant[V] |
| Part2 | q_norm[V] + q_b_quant[V] | kv_matmul[C] |
| Part3 | q_b_matmul[C] | kv_norm[V] + rope[V] + scatter[AIV] |
| Tail | q_rms[V] + rope[V]（等待 aux 完成） | — |

- 使用 `dsv4_dsa_overlap_stream()` 作为 aux stream。
- 通过 `npu_stream_switch` 和 `wait_stream`/`wait_event` 同步。
- 返回 `(q, qr, qr_pertoken_scale)`。

---

## 7. `_forward_prefill` 详细解析

```python
def _forward_prefill(
    self,
    layer_name,
    hidden_states: torch.Tensor,
    kv_cache: tuple[torch.Tensor, ...],
    attn_metadata: DSAMetadataList,
):
```

`_forward_prefill` 负责处理 prefill 阶段的 DSA 计算。输入 `hidden_states` 仅包含 prefill token，已经被 `forward` 从完整 batch 中切出。

### 7.1 解压 KV Cache

```python
(compress_kv_cache, swa_kv_cache, state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
    DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)
)
```

- `kv_cache` 是一个 tuple，根据 `compress_ratio` 不同包含不同数量的 tensor。
- `unpack_dsa_forward_kv_cache` 按固定顺序解压：
  - `compress_ratio == 4`：5 个元素 → `[attn, compressor.state_cache, indexer.compressor.state_cache, indexer.k_cache, swa_cache]`
  - `compress_ratio == 128`：3 个元素 → `[attn, compressor.state_cache, swa_cache]`
  - `compress_ratio <= 1`：1 个元素 → `[swa_cache]`
- 分别得到：
  - `compress_kv_cache`：压缩 KV cache（C4/C128 用）
  - `swa_kv_cache`：滑动窗口注意力 KV cache
  - `state_cache`：compressor 状态 cache
  - `indexer_k_cache` / `indexer_scale_cache` / `indexer_full_cache`：indexer 相关的 k/scale/full cache

### 7.2 选择对应的 Attention Metadata

```python
if self.compress_ratio == 4:
    (compressor_attn_metadata, compressor_kv_state_metadata, _, indexer_kv_scale_metadata, swa_metadata) = attn_metadata
    compress_common_attn_metadata = compressor_attn_metadata
elif self.compress_ratio == 128:
    (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata
    compress_common_attn_metadata = compressor_attn_metadata
else:
    (swa_metadata,) = attn_metadata
    compress_common_attn_metadata = swa_metadata
```

- `attn_metadata` 是一个 list，元素顺序与 KV cache 解压顺序一致。
- 提取出 compressor 的 attention metadata、compressor state metadata、indexer scale metadata、SWA metadata。
- `compress_common_attn_metadata` 是后续 sparse attention 使用的公共 metadata。

### 7.3 提取 Prefill 子 Metadata

```python
common_prefill_metadata = _require_prefill_metadata(compress_common_attn_metadata)
swa_prefill_metadata = _require_prefill_metadata(swa_metadata)
cos = common_prefill_metadata.cos[layer_name]
sin = common_prefill_metadata.sin[layer_name]
actual_seq_lengths_query = common_prefill_metadata.query_start_loc
actual_seq_lengths_key = common_prefill_metadata.seq_lens
```

- 从公共 metadata 和 SWA metadata 中提取 prefill 部分。
- `cos[layer_name]` / `sin[layer_name]` 是 dict，按 layer name 索引（因为不同 layer 的 rope 可能不同）。
- `actual_seq_lengths_query`：query 的累积长度，用于 sparse attention 算子。
- `actual_seq_lengths_key`：key 的序列长度。

### 7.4 生成 Query 和 KV（MLA Prolog）

#### 7.4.1 多流路径

```python
if self.multistream_dsv4_dsa_overlap:
    q, qr, _ = self._mla_prolog_multistream(
        hidden_states, cos, sin, swa_kv_cache, swa_prefill_metadata.slot_mapping, is_prefill=True
    )
```

- 调用 `_mla_prolog_multistream`。
- 在 main stream 上计算 q，在 aux stream 上并行计算 kv 并 scatter 到 `swa_kv_cache`。
- 返回 q、qr（q 投影中间结果）和 qr_pertoken_scale（prefill 下未使用）。

#### 7.4.2 单流路径

```python
else:
    share_hs_quant = _is_w8a8_dynamic(self.wq_a) and _is_w8a8_dynamic(self.wkv)
    if share_hs_quant:
        hs_int8, hs_pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
        q_a = torch_npu.npu_quant_matmul(...)
    else:
        q_a = self.wq_a(hidden_states)

    # q path
    if _is_w8a8_dynamic(self.wq_b):
        qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(q_a, self.q_norm.weight, epsilon=self.eps)
        q = torch_npu.npu_quant_matmul(qr, self.wq_b.weight, ...).unflatten(-1, (self.n_local_heads, self.head_dim))
    else:
        qr = self.q_norm(q_a)
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))

    q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)
    torch.ops._C_ascend.inplace_partial_rotary_mul(q.unsqueeze(1), cos, sin, ...)

    # kv path
    if share_hs_quant:
        kv = torch_npu.npu_quant_matmul(hs_int8, self.wkv.weight, ...)
    else:
        kv = self.wkv(hidden_states)
    kv = self.kv_norm(kv)
    kv = kv.view(-1, 1, self.nope_head_dim + self.rope_head_dim)
    torch.ops._C_ascend.inplace_partial_rotary_mul(kv.unsqueeze(1), cos, sin, ...)
    DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, swa_prefill_metadata.slot_mapping)
```

- **共享 hidden states 量化**：如果 `wq_a` 和 `wkv` 都是 W8A8 dynamic，则对 `hidden_states` 只做一次动态量化，复用给两个 matmul。
- **q 路径**：`wq_a` → `q_norm` → `wq_b` → reshape → `apply_dsa_q_rms` → partial rope。
- **kv 路径**：`wkv` → `kv_norm` → reshape → partial rope → `dsa_kv_compress_scatter` 写入 SWA KV cache。

### 7.5 无压缩路径（SWA Only）

```python
if self.compress_ratio <= 1:
    return attn_op(
        q,
        ori_kv=swa_kv_cache,
        ori_block_table=swa_prefill_metadata.block_table,
        cu_seqlens_q=actual_seq_lengths_query,
        seqused_kv=actual_seq_lengths_key,
        sinks=self.attn_sink,
        metadata=common_prefill_metadata.sas_metadata,
        softmax_scale=self.softmax_scale,
        cmp_ratio=max(self.compress_ratio, 1),
        ori_mask_mode=4,        # sliding window
        ori_win_left=self.window_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        **extra_attn_kwargs,
    )[0]
```

- 直接调用 `DeviceOperator.get_dsa_sparse_attn_op()`。
- 只使用原始 SWA KV，无需压缩 KV。
- `ori_mask_mode=4` 表示滑动窗口 mask。

### 7.6 有压缩路径（C4 / C128）

```python
if self.compress_ratio > 1:
    compressor_prefill_metadata = _require_prefill_metadata(compressor_attn_metadata)
    compressor_state_prefill_metadata = _require_prefill_metadata(compressor_kv_state_metadata)
```

#### 7.6.1 Indexer Top-K 计算（仅 compress_ratio == 4）

```python
if self.compress_ratio == 4:
    prefill_offset = attn_metadata[0].num_decode_tokens
    prefill_num_tokens = hidden_states.shape[0]
    if self.skip_topk:
        compress_topk_idxs = self._get_indexcache_topk_indices(prefill_num_tokens, offset=prefill_offset)
    else:
        if self.multistream_dsv4_dsa_overlap:
            indexer_q = self.cv_indexer_select_qli(...)
        else:
            compress_topk_idxs = self.indexer_select_qli(...)
```

- **C4 压缩需要 indexer** 来选择每个 query token 需要关注哪些压缩 KV 块。
- `prefill_offset`：因为 `forward` 中 prefill token 在 decode token 之后，index cache 偏移需要加上 decode token 数量。
- `skip_topk`：直接复用缓存的 top-k，不重新计算。
- 否则调用 `indexer_select_qli`（单流）或 `cv_indexer_select_qli`（多流）计算 top-k。
- 多流版本返回的是 `indexer_q`，调用者继续完成 `weights_proj` 和 lightning indexer。

#### 7.6.2 计算 Compressor 元数据

```python
coff = 2 if self.compressor_overlap else 1
compress_cos, compress_sin, compress_slot_mapping = self._compute_compressor_metadata(compressor_prefill_metadata)
```

- `coff`：compressor overlap 系数。
- 调用 `_compute_compressor_metadata` 获取压缩所需的 cos/sin/slot mapping。

#### 7.6.3 调用 Compressor 算子

```python
compressed_kv = torch.ops._C_ascend.compressor(
    hidden_states,
    self.compressor_wkv.weight,
    self.compressor_wgate.weight,
    state_cache.squeeze(-2),
    self.compressor_ape,
    self.compressor_norm.weight,
    compress_sin.view(-1, compress_sin.shape[-1]),
    compress_cos.view(-1, compress_cos.shape[-1]),
    state_block_table=compressor_state_prefill_metadata.block_table,
    cu_seqlens=actual_seq_lengths_query,
    seqused=None,
    start_pos=common_prefill_metadata.start_pos,
    rope_head_dim=self.rope_head_dim,
    cmp_ratio=self.compress_ratio,
    coff=coff,
    norm_eps=self.compressor_norm_eps,
    rotary_mode=2,
    cache_mode=1,
)
```

- 输入当前 `hidden_states`、compressor 的 weight、gate weight、state cache、ape、norm weight、rope cos/sin、state block table。
- 输出 `compressed_kv`，即压缩后的 KV。
- `cmp_ratio` 决定压缩倍率（4 或 128）。

#### 7.6.4 多流路径下的 Indexer 完成（C4）

```python
if self.multistream_dsv4_dsa_overlap and self.compress_ratio == 4 and not self.skip_topk:
    main_stream = torch.npu.current_stream()
    aux_stream = dsv4_dsa_overlap_stream()
    e_compressed_kv_done = main_stream.record_event()
    with npu_stream_switch(aux_stream, enabled=True):
        torch.npu.current_stream().wait_event(e_compressed_kv_done)
        weights_proj_output = self.weights_proj(hidden_states)
    q_quant, q_scale = DeviceOperator.indexer_quantize_query(indexer_q)
```

- aux stream 等待 main stream 的 compressor 完成。
- aux stream 并行计算 `weights_proj`。
- main stream 继续对 `indexer_q` 做量化。

#### 7.6.5 Scatter 压缩 KV

```python
if compressed_kv.shape[0] > 0:
    DeviceOperator.dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, compress_slot_mapping)
```

- 将 `compressed_kv` 写入 `compress_kv_cache`。
- 如果 compressor 输出为空（零行），则跳过 scatter。

#### 7.6.6 完成 Indexer（多流 C4）

```python
if self.multistream_dsv4_dsa_overlap and self.compress_ratio == 4 and not self.skip_topk:
    main_stream.wait_stream(aux_stream)
    weights = weights_proj_output * (self.indexer_softmax_scale * self.indexer_heads**-0.5)
    compress_topk_idxs, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(...)
```

- main stream 等待 aux stream 的 `weights_proj` 完成。
- 计算 weights scale。
- 调用 `npu_vllm_quant_lightning_indexer` 计算 top-k 索引。

#### 7.6.7 更新 Index Cache

```python
if self.compress_ratio == 4 and self.use_index_cache:
    self._update_indexcache_topk_indices(compress_topk_idxs, offset=prefill_offset)
```

- 如果启用 index cache，将当前层的 top-k 写入 buffer，供 `skip_topk` 层复用。

#### 7.6.8 调用 Sparse Attention

```python
if self.compress_ratio == 4:
    DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
        extra_attn_kwargs, cu_seqlens_cmp_kv=common_prefill_metadata.cu_c4_cmp_seqlen_list
    )
    attn_output = attn_op(
        q,
        ori_kv=swa_kv_cache,
        cmp_kv=compress_kv_cache,
        cmp_sparse_indices=compress_topk_idxs,
        ori_block_table=swa_prefill_metadata.block_table,
        cmp_block_table=compressor_prefill_metadata.block_table,
        cu_seqlens_q=actual_seq_lengths_query,
        seqused_kv=actual_seq_lengths_key,
        sinks=self.attn_sink,
        metadata=common_prefill_metadata.sas_metadata,
        softmax_scale=self.softmax_scale,
        cmp_ratio=self.compress_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,        # causal
        ori_win_left=self.window_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        **extra_attn_kwargs,
    )[0]
else:
    # compress_ratio == 128
    DeviceOperator.add_dsa_sparse_attn_extra_kwargs(
        extra_attn_kwargs, cu_seqlens_cmp_kv=common_prefill_metadata.cu_c128_cmp_seqlen_list
    )
    attn_output = attn_op(
        q,
        ori_kv=swa_kv_cache,
        cmp_kv=compress_kv_cache,
        ori_block_table=swa_prefill_metadata.block_table,
        cmp_block_table=compressor_prefill_metadata.block_table,
        ...
    )[0]
```

- C4 时传入 `cmp_sparse_indices`（top-k 索引），C128 时不传（C128 不需要 indexer）。
- `ori_mask_mode=4` 表示 SWA 滑动窗口；`cmp_mask_mode=3` 表示压缩 KV 使用 causal mask。
- 返回 `attn_output`。

---

## 8. `_forward_decode` 详细解析

```python
def _forward_decode(
    self,
    layer_name,
    hidden_states: torch.Tensor,
    kv_cache: tuple[torch.Tensor, ...],
    attn_metadata: DSAMetadataList,
):
```

`_forward_decode` 负责处理 decode 阶段的 DSA 计算。输入 `hidden_states` 仅包含 decode token，已经被 `forward` 从完整 batch 中切出。整体流程与 `_forward_prefill` 高度相似，但 metadata 和 slot mapping 使用 decode 版本。

### 8.1 解压 KV Cache

```python
(compress_kv_cache, swa_kv_cache, state_cache, indexer_k_cache, indexer_scale_cache, indexer_full_cache) = (
    DeviceOperator.unpack_dsa_forward_kv_cache(kv_cache, self.compress_ratio)
)
```

- 与 prefill 完全一致，根据 `compress_ratio` 解压出对应的 cache。

### 8.2 选择对应的 Attention Metadata

```python
if self.compress_ratio == 4:
    (compressor_attn_metadata, compressor_kv_state_metadata, _, indexer_kv_scale_metadata, swa_metadata) = attn_metadata
    compress_common_attn_metadata = compressor_attn_metadata
elif self.compress_ratio == 128:
    (compressor_attn_metadata, compressor_kv_state_metadata, swa_metadata) = attn_metadata
    compress_common_attn_metadata = compressor_attn_metadata
else:
    (swa_metadata,) = attn_metadata
    compress_common_attn_metadata = swa_metadata
```

- 与 prefill 一致。

### 8.3 提取 Decode 子 Metadata

```python
common_decode_metadata = _require_decode_metadata(compress_common_attn_metadata)
swa_decode_metadata = _require_decode_metadata(swa_metadata)
cos = common_decode_metadata.cos[layer_name]
sin = common_decode_metadata.sin[layer_name]
actual_seq_lengths_query = common_decode_metadata.query_start_loc
actual_seq_lengths_key = common_decode_metadata.seq_lens
```

- 使用 decode 版本的 metadata。
- decode 时每个请求通常只有 1 个 query token（或 spec decode 时多个）。

### 8.4 生成 Query 和 KV（MLA Prolog）

#### 8.4.1 多流路径

```python
if self.multistream_dsv4_dsa_overlap:
    q, qr, qr_pertoken_scale = self._mla_prolog_multistream(
        hidden_states, cos, sin, swa_kv_cache, swa_decode_metadata.slot_mapping, is_prefill=False
    )
```

- decode 路径下会返回 `qr_pertoken_scale`，因为 decode 的 indexer 可能复用 `qr_pertoken_scale` 做 W8A8 量化 matmul。

#### 8.4.2 单流路径

```python
else:
    share_hs_quant = _is_w8a8_dynamic(self.wq_a) and _is_w8a8_dynamic(self.wkv)
    if share_hs_quant:
        hs_int8, hs_pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)

    # q path
    if _is_w8a8_dynamic(self.wq_b):
        ...
        qr, qr_pertoken_scale = torch.ops._C_ascend.npu_rms_norm_dynamic_quant(...)
        q = torch_npu.npu_quant_matmul(qr, self.wq_b.weight, ...)
    else:
        ...
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        qr_pertoken_scale = None

    q = DeviceOperator.apply_dsa_q_rms(q, self.eps, self.q_norm_without_weight)
    torch.ops._C_ascend.inplace_partial_rotary_mul(q.unsqueeze(1), cos, sin, ...)

    # kv path
    ...
    DeviceOperator.dsa_kv_compress_scatter(swa_kv_cache, kv, swa_decode_metadata.slot_mapping)
```

- 与 prefill 单流路径基本一致。
- 区别：decode 的 `slot_mapping` 来自 `swa_decode_metadata`，需要经过 `DeviceOperator.pad_dsa_decode_slot_mapping` 处理。

### 8.5 压缩 KV 与 Indexer（C4 / C128）

#### 8.5.1 Indexer Top-K 计算（仅 compress_ratio == 4）

```python
if self.compress_ratio == 4:
    decode_num_tokens = hidden_states.shape[0]
    if self.skip_topk:
        compress_topk_idxs = self._get_indexcache_topk_indices(decode_num_tokens, offset=0)
    else:
        if self.multistream_dsv4_dsa_overlap:
            indexer_q = self.cv_indexer_select_qli(
                x=hidden_states,
                qr=qr,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                cos=cos,
                sin=sin,
                actual_seq_lengths_query=actual_seq_lengths_query,
                with_prefill=False,
                qr_pertoken_scale=qr_pertoken_scale,
            )
        else:
            compress_topk_idxs = self.indexer_select_qli(
                x=hidden_states,
                qr=qr,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                cos=cos,
                sin=sin,
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_key=actual_seq_lengths_key,
                with_prefill=False,
                qr_pertoken_scale=qr_pertoken_scale,
            )
```

- decode 的 `offset=0`，因为 decode token 在 index cache 的最前面。
- 多流版本返回 `indexer_q`；单流版本直接返回 `compress_topk_idxs`。

#### 8.5.2 Compressor 计算

```python
coff = 2 if self.compressor_overlap else 1
compress_cos, compress_sin, compress_slot_mapping = self._compute_compressor_metadata(compressor_decode_metadata)

compressed_kv = torch.ops._C_ascend.compressor(
    hidden_states,
    self.compressor_wkv.weight,
    self.compressor_wgate.weight,
    state_cache.squeeze(-2),
    self.compressor_ape,
    self.compressor_norm.weight,
    compress_sin.view(-1, compress_sin.shape[-1]),
    compress_cos.view(-1, compress_cos.shape[-1]),
    state_block_table=compressor_state_decode_metadata.block_table,
    cu_seqlens=actual_seq_lengths_query,
    seqused=None,
    start_pos=common_decode_metadata.start_pos,
    rope_head_dim=self.rope_head_dim,
    cmp_ratio=self.compress_ratio,
    coff=coff,
    norm_eps=self.compressor_norm_eps,
    rotary_mode=2,
    cache_mode=1,
)
```

- 与 prefill 一致，只是 metadata 换为 decode 版本。

#### 8.5.3 多流路径下的 Indexer 完成（C4）

```python
if self.multistream_dsv4_dsa_overlap and self.compress_ratio == 4 and not self.skip_topk:
    main_stream = torch.npu.current_stream()
    aux_stream = dsv4_dsa_overlap_stream()
    e_compressed_kv_done = main_stream.record_event()
    with npu_stream_switch(aux_stream, enabled=True):
        torch.npu.current_stream().wait_event(e_compressed_kv_done)
        weights_proj_output = self.weights_proj(hidden_states)
    q_quant, q_scale = DeviceOperator.indexer_quantize_query(indexer_q)
```

- 与 prefill 一致。

#### 8.5.4 Scatter 压缩 KV

```python
if compressed_kv.shape[0] > 0:
    DeviceOperator.dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, compress_slot_mapping)
```

- 与 prefill 一致。

#### 8.5.5 完成 Indexer（多流 C4）

```python
if self.multistream_dsv4_dsa_overlap and self.compress_ratio == 4 and not self.skip_topk:
    main_stream.wait_stream(aux_stream)
    weights = weights_proj_output * (self.indexer_softmax_scale * self.indexer_heads**-0.5)
    compress_topk_idxs, _ = torch.ops._C_ascend.npu_vllm_quant_lightning_indexer(...)
```

- 与 prefill 一致，使用 decode 版本的 metadata。

#### 8.5.6 更新 Index Cache

```python
if self.compress_ratio == 4 and self.use_index_cache:
    self._update_indexcache_topk_indices(compress_topk_idxs, offset=0)
```

- decode 的 offset 为 0。

### 8.6 调用 Sparse Attention

```python
attn_op = DeviceOperator.get_dsa_sparse_attn_op()
extra_attn_kwargs: dict = DeviceOperator.get_dsa_sparse_attn_base_kwargs()

if self.compress_ratio <= 1:
    attn_output = attn_op(
        q,
        ori_kv=swa_kv_cache,
        ori_block_table=swa_decode_metadata.block_table,
        cu_seqlens_q=actual_seq_lengths_query,
        seqused_kv=actual_seq_lengths_key,
        sinks=self.attn_sink,
        metadata=swa_decode_metadata.sas_metadata,
        ...
    )[0]
elif self.compress_ratio == 4:
    attn_output = attn_op(
        q,
        ori_kv=swa_kv_cache,
        cmp_kv=compress_kv_cache,
        cmp_sparse_indices=compress_topk_idxs,
        ori_block_table=swa_decode_metadata.block_table,
        cmp_block_table=compressor_decode_metadata.block_table,
        metadata=compressor_decode_metadata.sas_metadata,
        ...
    )[0]
else:
    attn_output = attn_op(
        q,
        ori_kv=swa_kv_cache,
        cmp_kv=compress_kv_cache,
        ori_block_table=swa_decode_metadata.block_table,
        cmp_block_table=compressor_decode_metadata.block_table,
        metadata=compressor_decode_metadata.sas_metadata,
        ...
    )[0]
```

- decode 的 sparse attention 与 prefill 参数一致。
- C4 时传入 `cmp_sparse_indices`，C128 时不传。
- 返回 `attn_output`。

---

## 9. Indexer 相关辅助函数

### 9.1 `_indexer_qkv_prepare(...)`

- 解压 indexer KV cache。
- 从 `qr` 计算 indexer q，应用 RoPE 和 Hadamard 旋转。
- 调用 `_C_ascend.compressor` 生成 indexer 的压缩 KV。
- 返回 q、kv、indexer k/scale/full cache、metadata、slot mapping 等。

### 9.2 `_indexer_qli_finish(...)`

- 调用 `_indexer_quant_scatter` 对 q/kv 做量化并 scatter。
- 调用 `_indexer_qli` 执行 lightning indexer。

### 9.3 `_indexer_quant_scatter(...)`

- 调用 `DeviceOperator.indexer_quant_scatter`。

### 9.4 `_indexer_qli(...)`

- 根据 `with_prefill` 选择 prefill/decode metadata。
- 调用 `_C_ascend.npu_vllm_quant_lightning_indexer` 计算 top-k 索引。

### 9.5 `indexer_select_qli(...)`

- 单流完整 indexer 流程：prepare → weights_proj → finish。

### 9.6 `cv_indexer_select_qli(...)`

- 多流 indexer 流程：
  - Part0：main stream 预计算 qr 量化、compressor、kv Hadamard。
  - Part1：main stream 做 q matmul，aux stream 并行做 kv 量化 + scatter k cache。
  - Part2：main stream 做 q RoPE。
  - Part3：main stream 做 q Hadamard linear，aux stream 并行 scatter scale cache。
  - 返回 q，由调用者继续完成 weights_proj、q 量化、lightning indexer。

---

## 10. 关键设计要点

1. **Multistream 重叠**：通过 `dsv4_dsa_overlap_stream()` 将 q 的 Cube 计算与 kv 的 Vector/AIV 操作并行，提高 NPU 利用率。
2. **静态 Buffer**：OTP 的 send/recv/reduce_scatter buffer 在第一次调用时懒分配，确保 ACL graph replay 地址稳定。
3. **IndexCache**：部分层可 `skip_topk` 复用前面层的 top-k，减少重复计算。
4. **Compressor 比例**：支持 `1`（纯 SWA）、`4`、`128` 三种压缩比，分别对应不同的 KV cache 结构和 indexer 行为。
5. **W8A8 动态量化**：在 q/kv 投影路径中自动检测并使用 `npu_quant_matmul` / `npu_rms_norm_dynamic_quant`。
6. **RoPE 处理**：q/kv 的 rope 在 prolog 中完成；o-proj 前做 inverse rope 恢复 nope 部分。
7. **Prefill/Decode 复用**：`_forward_prefill` 和 `_forward_decode` 逻辑几乎一致，差异主要在 metadata 类型和 offset 计算。

---

## 11. 运行脚本 `vllm_start_dp.sh` 实际分支分析

### 11.1 关键配置推导

运行脚本 `/home/w00608002/vllm-ascend/vllm_start_dp.sh` 的核心参数：

```bash
vllm serve /home/weight/DeepSeek-V4-Flash \
  --max_model_len 135000 \
  --max-num-batched-tokens 4096  \
  --enable-expert-parallel \
  --async-scheduling \
  --max-num-seqs 64 \
  --block-size 128 \
  --data-parallel-size 4 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --speculative-config '{"num_speculative_tokens": 1,"method": "deepseek_mtp", "enforce_eager": true}' \
  --additional_config '{"enable_cpu_binding": "True", "multistream_overlap_shared_expert": true}'
```

推导出的关键配置：

| 配置项 | 值 | 说明 |
|---|---|---|
| 模型 | `DeepSeek-V4-Flash` | `deepseek_v4` 模型类型 |
| 量化方式 | `fp8` / `deepseek_v4_fp8` | `quantization_config` 为 `fp8`（e4m3，dynamic） |
| 设备类型 | **A5 (Ascend950)** | `_build_info.__device_type__ = 'A5'`，`npu-smi` 显示 `Ascend950DT` |
| `tensor_parallel_size` | `1` | 未显式设置 `--tensor-parallel-size` |
| `data_parallel_size` | `4` | 由 `--data-parallel-size 4` |
| `enable_expert_parallel` | `True` | `--enable-expert-parallel` |
| `cudagraph_mode` | `FULL_DECODE_ONLY` | decode 阶段走 ACL graph |
| `multistream_dsv4_dsa_overlap` | `True`（默认） | `additional_config` 未覆盖，默认 `True` |
| `oproj_tensor_parallel_size` | `0`（默认） | `additional_config` 未设置 `finegrained_tp_config` |
| `olora_tensor_parallel_size` | `0`（默认） | 同上 |
| `use_index_cache` | `None` | 模型 `config.json` 中未设置 |
| `block_size` | `128` | `--block-size 128` |

---

### 11.2 `AscendDSAMetadataBuilder` 中走的分支

#### 11.2.1 `__init__`

- `slot_mapping_shape = (max_num_batched_tokens,)`（A5 设备特殊处理，非 A5 为 `(max_num_batched_tokens, 2)`）。
- `speculative_config` 存在（`num_speculative_tokens=1`），会初始化 `spec_slot_mapping[0]` 和 `spec_sas_metadata[0]`。
- 模型类型为 `deepseek_v4`，会生成类级 `hadamard` 矩阵。

#### 11.2.2 `build`

- `decode_threshold` 初始为 1，因存在 speculative decoding，增加到 `1 + 1 = 2`。
- 调用 `split_decodes_and_prefills(..., decode_threshold=2)` 划分 decode/prefill。
- 由于 `cudagraph_mode = FULL_DECODE_ONLY`，实际运行时 decode 阶段会构建 graph metadata，prefill 阶段不走 graph。

#### 11.2.3 `build_prefill_metadata` / `build_decode_metadata`

根据模型 `config.json` 中的 `compress_ratios` 数组，每层 `compress_ratio` 不同。DeepSeek-V4-Flash 的 `compress_ratios` 为：

```python
[0, 0, 4, 128, 4, 128, 4, 128, ..., 4, 0]
```

因此：

- **Layer 0, 1, 42**：`compress_ratio <= 1` → SWA-only。
- **Layer 2, 4, 6, ..., 40**（c4 层）：`compress_ratio = 4` → C4 + indexer。
- **Layer 3, 5, 7, ..., 41**（c128 层）：`compress_ratio = 128` → C128，无 indexer。

#### 11.2.4 `use_index_cache` 和 `skip_topk`

- `config.json` 中 `use_index_cache` 为 `None`，所以所有 c4 层的 `skip_topk = False`。
- **每个 c4 层都会独立计算 top-k**，不会复用其他层的 index cache。

---

### 11.3 `AscendDSAImpl` 中 `_forward_prefill` / `_forward_decode` 走的分支

#### 11.3.1 顶层 `forward`

- 因 `cudagraph_mode = FULL_DECODE_ONLY`，decode 阶段 `attn_state = AscendAttentionState.DecodeOnly`（或 `SpecDecoding`），prefill 阶段为 `ChunkedPrefill`。
- `forward` 中会拆分 decode 和 prefill token，分别调用 `_forward_decode` 和 `_forward_prefill`。
- 由于存在 speculative decoding（MTP），draft iteration 会走 `build_for_drafting` 路径，但限制 `compress_ratio <= 1`，所以 draft attention 只走 SWA-only。

#### 11.3.2 MLA Prolog 分支

在 `_forward_prefill` 和 `_forward_decode` 中：

```python
if self.multistream_dsv4_dsa_overlap:
    q, qr, qr_pertoken_scale = self._mla_prolog_multistream(...)
else:
    # 单流路径
    ...
```

由于 `multistream_dsv4_dsa_overlap = True`（默认），**实际走多流路径 `_mla_prolog_multistream`**。

#### 11.3.3 W8A8 动态量化分支

`_mla_prolog_multistream` 中：

```python
is_w8a8 = _is_w8a8_dynamic(self.wq_b)
```

- 当前模型是 **FP8 (e4m3) 量化**，对应的 scheme 是 `AscendW8A8MXFP8DSDynamicLinearMethod`（`FP8` scheme），它继承自 `AscendW8A8MXFP8DynamicLinearMethod`。
- `_is_w8a8_dynamic()` 只检查是否是 `AscendW8A8DynamicLinearMethod` 实例（`W8A8_DYNAMIC` scheme），**FP8 scheme 不是这个类型**。
- 因此 `_is_w8a8_dynamic(self.wq_b)` 返回 `False`。

但是在 `_mla_prolog_multistream` 中：

```python
q_quant, q_pertoken_scale = self.cv_wq_a.quantize(hidden_states)
```

- `CVLinearWrapper.quantize()` 检测 `_is_w8a8_dynamic` 时，同样只检测 `AscendW8A8DynamicLinearMethod`。
- 对于 FP8 量化层，`cv_wq_a.quantize()` 不会走 `npu_dynamic_quant`，而是返回 `(x, None)`。
- 然后 `cv_wq_a.matmul(q_quant, q_pertoken_scale)` 会调用 `linear.quant_method.apply(self.linear, quantized_x, bias)`，即 `AscendW8A8MXFP8DSDynamicLinearMethod.apply()`，内部会走 `npu_dynamic_mx_quant` + `npu_quant_matmul` 的 FP8 路径。

**结论**：`_mla_prolog_multistream` 走多流路径，但 q/kv 的量化 matmul 最终由 `CVLinearWrapper` 委托给 FP8 scheme 处理，不是 `_is_w8a8_dynamic` 分支中的 `npu_quant_matmul`（int8 weight）。

#### 11.3.4 Compressor 与 Indexer 分支

##### `compress_ratio <= 1` 的层（SWA-only）

```python
if self.compress_ratio <= 1:
    return attn_op(
        q,
        ori_kv=swa_kv_cache,
        ori_block_table=swa_prefill_metadata.block_table,  # 或 swa_decode_metadata
        ...
        cmp_ratio=1,
        ori_mask_mode=4,        # sliding window
        ...
    )[0]
```

**直接走 SWA-only 稀疏 attention，无压缩 KV、无 indexer。**

##### `compress_ratio = 4` 的层（C4）

```python
if self.compress_ratio == 4:
    # 计算 indexer top-k
    if self.skip_topk:
        compress_topk_idxs = self._get_indexcache_topk_indices(...)
    else:
        if self.multistream_dsv4_dsa_overlap:
            indexer_q = self.cv_indexer_select_qli(...)   # 多流 indexer
        else:
            compress_topk_idxs = self.indexer_select_qli(...)  # 单流 indexer

    # 调用 compressor
    compressed_kv = torch.ops._C_ascend.compressor(...)

    # 多流完成 indexer
    if self.multistream_dsv4_dsa_overlap and self.compress_ratio == 4 and not self.skip_topk:
        ...

    # scatter 压缩 KV
    DeviceOperator.dsa_kv_compress_scatter(compress_kv_cache, compressed_kv, compress_slot_mapping)

    # sparse attention
    attn_output = attn_op(
        q,
        ori_kv=swa_kv_cache,
        cmp_kv=compress_kv_cache,
        cmp_sparse_indices=compress_topk_idxs,   # 仅 C4 需要
        ...
        cmp_ratio=4,
        ...
    )[0]
```

由于 `skip_topk = False` 且 `multistream_dsv4_dsa_overlap = True`：

- **多流路径下**：`cv_indexer_select_qli` 在 aux stream 并行计算 indexer 的 kv scatter，main stream 计算 q，最后调用 `npu_vllm_quant_lightning_indexer` 得到 `compress_topk_idxs`。
- 然后调用 compressor 生成 C4 压缩 KV，scatter 到 `compress_kv_cache`。
- 最后调用 sparse attention，传入 `ori_kv`（SWA）+ `cmp_kv`（C4）+ `cmp_sparse_indices`。

##### `compress_ratio = 128` 的层（C128）

```python
elif self.compress_ratio == 128:
    attn_output = attn_op(
        q,
        ori_kv=swa_kv_cache,
        cmp_kv=compress_kv_cache,
        ori_block_table=...,
        cmp_block_table=...,
        ...
        cmp_ratio=128,
        ...
    )[0]
```

**调用 compressor 生成 C128 压缩 KV，但不传入 `cmp_sparse_indices`**（C128 不需要 indexer）。

#### 11.3.5 `_forward_o_proj` 分支

在 `forward` 最后：

```python
self._forward_o_proj(o_proj_input, output)
```

根据配置判断：

```python
if get_ascend_device_type() in {AscendDeviceType.A5}:
    # A5 FP8 量化 o-proj 路径
    o, swiglu_out_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
    o = torch_npu.npu_transpose_quant_batchmatmul(...)
    output[...] = self.wo_b(o)
elif oproj_tp_enable():
    # OTP 路径
    ...
elif olora_tp_enable():
    # o_lora TP 路径
    ...
else:
    # 默认路径
    ...
```

由于：
- 设备是 **A5**
- `oproj_tp_enable() = False`（未设置 `oproj_tensor_parallel_size`）
- `olora_tp_enable() = False`

**所以 `_forward_o_proj` 走 A5 FP8 量化路径**：`npu_dynamic_mx_quant` + `npu_transpose_quant_batchmatmul` + `wo_b`。

---

### 11.4 实际走的分支总结

| 场景 | 分支 |
|---|---|
| 设备类型 | **A5 (Ascend950)** |
| `multistream_dsv4_dsa_overlap` | **True** → 多流 MLA prolog |
| `compress_ratio <= 1` 层 | **SWA-only**，无压缩、无 indexer |
| `compress_ratio = 4` 层 | **C4 压缩 + 多流 indexer (`cv_indexer_select_qli`) + sparse attention with `cmp_sparse_indices`** |
| `compress_ratio = 128` 层 | **C128 压缩 + sparse attention without `cmp_sparse_indices`** |
| `skip_topk` | **False**（`use_index_cache` 未设置） |
| `_is_w8a8_dynamic` | **False**（FP8 scheme 不是 W8A8_DYNAMIC） |
| q/kv matmul | 由 `CVLinearWrapper` 委托给 FP8 scheme 的 `npu_dynamic_mx_quant` + `npu_quant_matmul` |
| `_forward_o_proj` | **A5 FP8 路径**：`npu_dynamic_mx_quant` → `npu_transpose_quant_batchmatmul` → `wo_b` |
| `oproj_tp_enable` | **False** |
| `olora_tp_enable` | **False** |

---

## 12. 实验性 Q/KV 伪量化实现

在 `vllm_ascend/attention/dsa_v1.py` 中添加了可选的伪量化路径，用于在 DSA sparse attention 调用前对 `q` 和 `kv` 注入量化噪声，以便快速评估低精度 Q/KV 对模型精度的影响。

> **注意**：这是**伪量化**（fake quantization）。输出张量仍然是 bf16，不会节省显存或带宽，也不会产生真实的量化打包数据。下游 attention kernel 无需任何修改即可消费这些张量。

### 12.1 新增模块 `vllm_ascend/attention/pseudo_quant.py`

#### 12.1.1 `pseudo_quantize_fp4_per_block(x, block_size=32, scale_dtype=torch.bfloat16)`

对输入张量最后一维按 `block_size` 做 per-block FP4 伪量化。实现已改为图模式友好：

1. 不再使用 `if x.shape[-1] % block_size != 0: raise ValueError` 这类 Python 控制流。
2. 通过 `_pad_to_multiple` 自动将最后一维零补齐到 `block_size` 的倍数；当余数为 0 时，`F.pad` 的 pad 列表全为 0，等价 no-op。
3. 将补齐后的张量 reshape 为 `[-1, num_blocks, block_size]`。
4. 每块计算 scale：`scale = max(|x|) / 7.0`，存为 bf16。
5. 将数值量化到 FP4 对称范围 `[-8, 7]`，再反量化为原始 dtype。
6. 使用静态 slice 截断回原始最后一维长度，恢复原始形状。

#### 12.1.2 `pseudo_quantize_hif8_per_tensor_fixed_scale(x, scale=8.0)`

对输入张量做 **per-tensor HiFloat8 (HiF8)** 伪量化，使用**固定 scale**，默认 `scale=8.0`。底层实现为 `_quant_hif8`，无 Python `if` 分支，图模式/TorchScript/ACL graph 兼容：

1. 将输入转换为 fp32，除以固定 scale：`x_scaled = x / scale`。
2. 对 `x_scaled` 做原始 HiF8 round-to-nearest 量化：
   - 计算 `e = floor(log2(|x| + eps))`，其中 `eps = 2**-45`（已统一为常量，无 dtype 分支）。
   - 根据 `|e|` 确定尾数位宽：
     - `|e| <= 3`：3 bit 尾数
     - `|e| <= 7`：2 bit 尾数
     - `|e| <= 15`：1 bit 尾数
     - 其他：0 bit 尾数
   - 按尾数位宽做 floor-round，再还原到原始指数位置。
   - 保留原符号。
3. 将量化后的值乘以 scale，反量化回原始 dtype。

所有分支均通过 `torch.where` 表达，无数据依赖的 Python `if` 控制流。

> 该函数用于当前 DSA 的 Q 伪量化。原始 `pseudo_quantize_hif8_per_token`（per-token 动态 scale）保留但不再用于 DSA 主路径。

#### 12.1.3 `pseudo_quantize_qkv_for_dsa(q, kv, kv_block_size=32, ...)`

组合上述两个函数：

- `q` 走 `pseudo_quantize_hif8_per_token`（per-token HiFP8）。
- `kv` 走 `pseudo_quantize_fp4_per_block`（per-block FP4，block size 32）。
- 返回伪量化后的 `(q, kv)`。

> 该组合函数当前未在 DSA 主路径中使用；DSA 主路径已改为分别调用 Q/KV 量化函数。

### 12.2 `dsa_v1.py` 中的集成点

#### 12.2.1 配置开关

在 `AscendDSAImpl.__init__` 中读取 `AscendConfig` 配置：

```python
self.enable_qkv_pseudo_quant = bool(
    getattr(ascend_config, "enable_qkv_pseudo_quant", False)
)
self.kv_pseudo_quant_block_size = int(
    getattr(ascend_config, "kv_pseudo_quant_block_size", 32)
)
```

- `enable_qkv_pseudo_quant`：是否启用伪量化。
- `kv_pseudo_quant_block_size`：KV FP4 per-block 的块大小，默认 32。

注意：`AscendConfig` 必须同步从 `additional_config` 中解析这两个字段，否则开关永远不会生效。

#### 12.2.2 `_forward_prefill` 中的插入点

在 SWA KV scatter 完成后、sparse attention 调用前插入对 **Q** 的伪量化：

```python
# Experimental: pseudo-quantize Q before sparse attention. The output
# remains bf16 so the existing attention kernel needs no change.
if self.enable_qkv_pseudo_quant:
    q = pseudo_quantize_hif8_per_tensor_fixed_scale(q, scale=8.0)
```

在 compressor scatter 完成后，对**压缩 KV** 做伪量化：

```python
# A zero-row compressor output has no KV writes. Skip scatter
# instead of passing None; A5 scatter dereferences x.view().
if compressed_kv.shape[0] > 0:
    # Experimental: pseudo-quantize compressed KV before writing it
    # to the cache so later reads observe the quantized values.
    if self.enable_qkv_pseudo_quant:
        compressed_kv = pseudo_quantize_fp4_per_block(
            compressed_kv, block_size=self.kv_pseudo_quant_block_size
        )
    DeviceOperator.dsa_kv_compress_scatter(
        compress_kv_cache, compressed_kv, compress_slot_mapping
    )
```

#### 12.2.3 `_forward_decode` 中的插入点

与 `_forward_prefill` 对称：

- decode SWA KV scatter 后，对 **Q** 做 HiF8 per-tensor 固定 scale 伪量化。
- compressor scatter 后，对 **compressed_kv** 做 FP4 per-block 伪量化。

#### 12.2.4 不量化的部分

- **SWA KV 不再伪量化**：当前实现已删除对 `swa_kv_cache` 的 FP4 伪量化，仅保留对 `compressed_kv` 的伪量化。
- **`_mla_prolog_multistream` 内部不再伪量化 Q**：Q 的伪量化统一在外层 `_forward_prefill` / `_forward_decode` 中完成，避免多流路径下重复量化。

### 12.3 启用方式

在启动命令的 `--additional_config` 中增加：

```bash
--additional_config '{
  "enable_cpu_binding": "True",
  "multistream_overlap_shared_expert": true,
  "enable_qkv_pseudo_quant": true,
  "kv_pseudo_quant_block_size": 32
}'
```

### 12.4 限制与未来工作

1. **伪量化性质**：当前输出仍为 bf16，仅用于精度验证。若要接入真实低精度 kernel，需要：
   - 将 KV cache 改为真实的 FP4 packed dtype（如 `torch.quint4x2` 或 `torch_npu.float4_e2m1fn_x2`）。
   - 修改 `DeviceOperator.get_dsa_sparse_attn_op()` 对应的 `kv_quant_mode` / `query_quant_mode` 等参数。
   - 相应修改 metadata builder 以传递量化相关的 layout 和 scale 信息。
2. **scale 精度**：压缩 KV 的 scale 以 bf16 保存，符合需求；量化/反量化的中间计算使用 fp32 保证数值稳定性。Q 的 HiF8 使用固定 scale=8.0。
3. **未经验证**：当前代码仅通过 Python 编译检查，未在真实 NPU 上运行验证精度和性能。
4. **全局开关**：当前 `enable_qkv_pseudo_quant` 对所有 DSA 层生效。若需按层控制，可扩展为 per-layer 配置。
5. **图模式友好性**：`pseudo_quant.py` 已去掉数据依赖的 Python `if` 分支，使用 `F.pad`、静态 slice、`torch.where` 等算子实现，便于被 `torch.compile` / torchair / ACL graph 捕获。对于 DeepSeek-V4-Flash 的固定 `head_dim=128`，`block_size=32` 恰好整除，不会触发 padding 分支。

