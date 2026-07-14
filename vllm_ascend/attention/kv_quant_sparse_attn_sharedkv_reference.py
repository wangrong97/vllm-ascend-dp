"""PyTorch reference implementation of ``kv_quant_sparse_attn_sharedkv``.

This is a functional, CPU/GPU-runnable reference for the Ascend custom operator
``kv_quant_sparse_attn_sharedkv`` (CANN arch35 / Ascend 950) used by the DSA
(Dynamic Sparse Attention) v1 backend in ``vllm_ascend/attention/dsa_v1.py``.

The NPU kernel is an AIC(Cube)+AIV(Vector) hybrid that fuses KV dequantization,
PageAttention gather, and FlashAttention. This module reproduces its *numerical
semantics* only; it is intentionally simple and un-optimized.

KV dtype / dequantization is MANDATORY (not optional). The host-side check
(``kv_quant_sparse_attn_sharedkv_check_single_para.cpp``) only allows
``ori_kv``/``cmp_kv`` to be ``INT8`` or ``FLOAT8_E4M3FN``, forces the last dim
to 640 (packed bytes), and forces ``kv_quant_mode=1``/``tile_size=64``. A bf16 KV
is rejected at compile time. The kernel therefore always runs ``DequantKv``
(fp8 -> bf16) before ``Q@K^T``.

What actually arrives at the op on Ascend A5 (Ascend950, arch35): the SWA /
compressor KV caches are allocated as ``float8_e4m3fn`` with ``cached_head_size
= head_dim + 128 = 640`` (see ``AscendDeepseekV4SWACache.get_kv_cache_spec`` in
``vllm_ascend/models/deepseek_v4.py``). bf16 KV is produced by the model, then
``A5DeviceAdaptor.dsa_kv_compress_scatter`` calls ``kv_compress_epilog``
(``quant_mode=2``, ``quant_group_size=64``) which fuses quantize + compress +
scatter, so the cache that the attention op reads is the 640B fp8-packed buffer.
(Non-A5 devices allocate the cache as ``int8`` instead.) On the Python side we
represent that buffer as ``uint8`` and ``view`` it to fp8 dtypes — all three are
1-byte storage and numerically equivalent.

Key semantics reproduced (verified against
``csrc/attention/kv_quant_sparse_attn_sharedkv``):

* PageAttention-style block-table KV gathering (``ori_block_table`` / ``cmp_block_table``).
* FP8-E4M3 packed KV dequantization with per-tile (64-dim) FP8-E8M0 scales.
  Packed layout = ``[kv_rope(64 bf16) | kv_nope(448 fp8_e4m3) | nope_scale(7 fp8_e8m0) | pad]``
  (640 bytes); dequantized KV is laid out ``[nope(448) | rope(64)]`` (D=512).
* Band / sliding-window mask (``ori_mask_mode=4``) over ``ori_kv``.
* Right-down causal mask (``cmp_mask_mode=3``) over ``cmp_kv`` for ``cmp_ratio=128``.
* Sparse top-k attention (``cmp_ratio=4``) where ``cmp_sparse_indices`` holds the
  per-query compressed-KV *logical token indices*; indices are filtered by a
  causal bound before being gathered.
* Shared KV: the dequantized KV is used simultaneously as K and V.
* Learnable attention sinks (``sinks``, shape ``[N_q]``): a per-head scalar logit
  that enters the softmax denominator only (no associated value).

The three compute modes are selected by ``cmp_ratio``:
  * ``1``   : Sliding-window attention only (SWA).
  * ``128`` : SWA + dense compressed attention (CFA), right-down causal.
  * ``4``   : SWA + sparse compressed attention (SCFA), per-query top-k.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from vllm_ascend.attention.pseudo_quant import (
    pseudo_quantize_hif8_per_tensor_fixed_scale,
)


# ---------------------------------------------------------------------------
# FP8 dequantization helpers
# ---------------------------------------------------------------------------

# FP8-E4M3 has a 1.0-representable range; FP8-E8M0 is a pure power-of-2 scale.
# torch natively understands these dtypes, so we just view-and-cast.
_FP8_E4M3 = torch.float8_e4m3fn
_FP8_E8M0 = torch.float8_e8m0fnu


def fp8_e4m3_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """Dequantize a uint8 buffer holding fp8_e4m3fn values into fp32."""
    if x.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 buffer, got {x.dtype}")
    return x.view(_FP8_E4M3).to(torch.float32)


def fp8_e8m0_to_fp32(x: torch.Tensor) -> torch.Tensor:
    """Dequantize a uint8 buffer holding fp8_e8m0fnu values into fp32.

    FP8-E8M0 is a pure-exponent format: byte ``b`` (0-255) encodes the value
    ``2^(b - 127)`` with ``b == 0`` mapping to 0. As an IEEE-754 float32 the
    same value has exponent field ``b`` (since ``(b-127) + 127 == b``) and a
    zero mantissa, i.e. bit pattern ``b << 23``. We synthesize that pattern with
    uint8 bit arithmetic and reinterpret as float32 instead of casting through
    ``float8_e8m0fnu``: on NPU that cast routes through ``aclnnInplaceCopy``
    which fails with error 561103 (e8m0 -> fp32 is not a supported cast).
    """
    if x.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 buffer, got {x.dtype}")
    bits = x.to(torch.int32) << 23
    return bits.view(torch.float32)


def dequant_packed_kv(
    kv_packed: torch.Tensor,
    nope_head_dim: int = 448,
    rope_head_dim: int = 64,
    tile_size: int = 64,
) -> torch.Tensor:
    """Dequantize packed KV tokens.

    The 640-byte packed layout used by the kernel is::

        [kv_rope(64 bf16 dims = 128B)
         kv_nope(448 fp8_e4m3 dims = 448B)
         nope_quant_scale(7 fp8_e8m0 dims = 7B)
         pad(57B to reach 640B)]

    Args:
        kv_packed: shape ``[..., 640]``, dtype uint8 (fp8_e4m3fn storage).
        nope_head_dim: Non-RoPE KV dim. Kernel-fixed at 448.
        rope_head_dim: RoPE KV dim. Kernel-fixed at 64.
        tile_size: Quantization granularity for nope scales. Kernel-fixed at 64.

    Returns:
        Dequantized KV of shape ``[..., 512]``, dtype bfloat16, laid out as
        ``[kv_nope(448) | kv_rope(64)]`` to match the kernel-internal format
        (see ``DequantKv``: *dstTensor是nope(448) + rope(64)*).
    """
    if kv_packed.shape[-1] != 640:
        raise ValueError(f"Expected last dim 640, got {kv_packed.shape[-1]}")

    rope_bytes = rope_head_dim * 2  # bf16
    nope_bytes = nope_head_dim      # one fp8 byte per dim
    scale_bytes = nope_head_dim // tile_size  # 448 / 64 = 7

    kv_rope_u8 = kv_packed[..., :rope_bytes]
    kv_nope_u8 = kv_packed[..., rope_bytes : rope_bytes + nope_bytes]
    scale_u8 = kv_packed[
        ..., rope_bytes + nope_bytes : rope_bytes + nope_bytes + scale_bytes
    ]

    # rope is stored as raw bf16 bytes.
    kv_rope = kv_rope_u8.view(torch.bfloat16)

    # nope: fp8_e4m3 -> fp32, then reshape to [..., num_tiles, tile_size].
    kv_nope_fp32 = fp8_e4m3_to_fp32(kv_nope_u8)
    kv_nope_fp32 = kv_nope_fp32.view(*kv_nope_fp32.shape[:-1], scale_bytes, tile_size)

    # scale: fp8_e8m0 -> fp32, broadcast over each 64-dim tile.
    scale_fp32 = fp8_e8m0_to_fp32(scale_u8).unsqueeze(-1)

    kv_nope_dequant = (kv_nope_fp32 * scale_fp32).to(torch.bfloat16)
    kv_nope_dequant = kv_nope_dequant.view(
        *kv_nope_dequant.shape[:-2], nope_head_dim
    )

    # Kernel-internal layout is [nope | rope]; K and V share this tensor.
    return torch.cat([kv_nope_dequant, kv_rope], dim=-1)


# ---------------------------------------------------------------------------
# PageAttention KV gathering
# ---------------------------------------------------------------------------

def gather_page_attention_kv(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Gather KV tokens from a page-attention block cache.

    Mirrors ``GetkeyOffset`` / ``CopyInKvNotSparse`` in the kernel: a logical
    token index ``s2Idx`` maps to physical block ``block_table[b, s2Idx // B]``
    at intra-block offset ``s2Idx % B``.

    Args:
        kv_cache: shape ``[num_blocks, block_size, KV_N, D_pack]`` (KV_N=1).
        block_table: shape ``[B, max_blocks]``. Only ``block_table[0]`` is used;
            gather per-batch and concatenate if B > 1.
        positions: logical token indices to gather, any shape.

    Returns:
        Gathered KV of shape ``positions.shape + [KV_N, D_pack]``.
    """
    num_blocks, block_size, kv_n, d_pack = kv_cache.shape
    if block_table.shape[0] != 1:
        raise NotImplementedError(
            "gather_page_attention_kv only supports a single batch row in "
            "block_table; call per-batch and concatenate."
        )

    flat_kv = kv_cache.reshape(-1, kv_n, d_pack)
    block_ids = positions // block_size
    block_offsets = positions % block_size
    physical_block_ids = block_table[0][block_ids]
    flat_indices = physical_block_ids * block_size + block_offsets
    flat_indices_1d = flat_indices.reshape(-1).to(torch.int64)

    # The downstream dequant pipeline (``dequant_packed_kv``) treats the 640-byte
    # packed KV as a ``uint8`` buffer (rope -> view(bf16), nope/scale ->
    # fp8_e4m3/e8m0 decode), so this gather must always return uint8 regardless
    # of the on-device cache dtype (float8_e4m3fn on A5, int8 elsewhere — all
    # 1-byte storage). On NPU, indexing those 1-byte dtypes directly fails with
    # aclnnIndex(Index)Select error 161002, so reinterpret the buffer as int16
    # (a 2-byte view; the row byte length kv_n*d_pack*element_size() is even for
    # d_pack=640), gather, then view the bytes as uint8. Pure byte relocation —
    # numerically identical to indexing the original tensor.
    row_bytes = kv_n * d_pack * flat_kv.element_size()
    assert row_bytes % 2 == 0, f"row is not 2-byte aligned: {row_bytes} bytes"
    flat_kv_view = flat_kv.view(torch.int16) if flat_kv.dtype != torch.int16 \
        else flat_kv
    gathered_view = torch.index_select(flat_kv_view, 0, flat_indices_1d)
    gathered_u8 = gathered_view.view(torch.uint8)
    return gathered_u8.reshape(*positions.shape, kv_n, d_pack)


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_band_mask(
    q_positions: torch.Tensor,
    kv_positions: torch.Tensor,
    win_left: int = 127,
    win_right: int = 0,
) -> torch.Tensor:
    """Boolean mask (True = masked out) for ``ori_mask_mode=4`` (band/SWA).

    ``q_positions`` is the query token's anchor on the *KV axis* (i.e.
    ``q_pos + nextTokensPerBatch`` where ``nextTokensPerBatch = S2 - S1``, see
    ``GetSingleCoreParam``). A query anchored at KV position ``i`` attends to
    ori_kv position ``j`` iff ``i - win_left <= j <= i + win_right``. For pure
    prefill with S2 == S1 this reduces to the usual ``[i-127, i]`` window.
    """
    q_pos = q_positions.unsqueeze(-1)  # [..., T, 1]
    kv_pos = kv_positions.unsqueeze(-2)  # [..., 1, S]
    return (kv_pos < q_pos - win_left) | (kv_pos > q_pos + win_right)


def build_right_down_causal_mask(
    q_positions: torch.Tensor,
    kv_positions: torch.Tensor,
    cmp_ratio: int,
) -> torch.Tensor:
    """Boolean mask (True = masked out) for ``cmp_mask_mode=3`` (right-down causal).

    ``q_positions`` is the query token's anchor on the *KV axis*
    (``q_pos + nextTokensPerBatch``). The kernel computes the compressed-KV
    window as an *exclusive* upper bound
    ``s2CmpLineEndIdx = s2LineEndIdx // cmpRatio`` where
    ``s2LineEndIdx = q_kv_pos + 1`` (the ``+s1RealSize`` term with ``s1RealSize=1``
    per query row, see ``ComputeS2LoopInfo``). A cmp position ``j`` therefore
    participates iff ``j < (q_kv_pos + 1) // cmp_ratio`` (equivalently
    ``j <= (q_kv_pos + 1) // cmp_ratio - 1``). This is also the bound enforced
    per-token in the SCFA path (``CopyInKvSparse`` ``s2IdLimit``).
    """
    q_pos = q_positions.unsqueeze(-1)
    kv_pos = kv_positions.unsqueeze(-2)
    # Exclusive upper bound on the KV axis, then compress.
    cmp_exclusive_end = (q_pos + 1) // cmp_ratio  # [..., T, 1]
    return kv_pos >= cmp_exclusive_end


# ---------------------------------------------------------------------------
# Core attention primitives (with learnable sink)
# ---------------------------------------------------------------------------

def _flash_attention_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    mask: Optional[torch.Tensor] = None,
    sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Plain attention ``softmax(Q K^T scale) V`` with an optional learnable sink.

    Args:
        q: ``[T, N, D]``.
        k: ``[S, D]`` (shared across heads).
        v: ``[S, D]`` (shared across heads).
        scale: softmax scale.
        mask: ``[T, S]`` boolean, True = do not attend.
        sink: ``[N]`` float32 learnable sink logit, broadcast across query tokens.
            Following the kernel's Flash init (``m0 = sink``, ``s0 = 1``), the
            running max is seeded with ``sink`` and the running sum with
            ``exp(sink - m0) = 1``; subsequent real-KV blocks update both via the
            standard online-softmax rule. Numerically this is equivalent to
            ``m = max(max(scores), sink)`` and ``Z = exp(sink - m) + Σ exp(score - m)``
            — i.e. sink is a virtual logit in the softmax denominator with no
            associated value.

    Returns:
        ``[T, N, D]``.
    """
    # [T, N, S]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

    scores_f32 = scores.to(torch.float32)
    # Running max over the real KV dim; if a sink is present it seeds the Flash
    # max (m0 = sink), so take the joint maximum (NOT clamped at 0 — the kernel
    # does not clamp, and a clamp would change results when the sink dominates).
    real_max = scores_f32.amax(dim=-1)  # [T, N]
    if sink is not None:
        # -inf rows (fully masked) collapse: max(-inf, sink) = sink.
        rowmax = torch.maximum(real_max, sink.to(torch.float32).unsqueeze(0))
    else:
        rowmax = real_max
    rowmax = rowmax.unsqueeze(-1)  # [T, N, 1]

    exp_scores = torch.exp(scores_f32 - rowmax)  # [T, N, S]
    # Pseudo-quantize the (un-normalized) exp scores to HiF8 per-tensor with a
    # fixed scale, matching the A8C4 path's exp-score quantization. The round
    # trip is in float32; dtype/shape are preserved by the helper.
    exp_scores = pseudo_quantize_hif8_per_tensor_fixed_scale(
        exp_scores, scale=8.0
    )  # [T, N, S]
    # Denominator = sink term (broadcast over T) + sum of real exp scores.
    if sink is not None:
        sink_term = torch.exp(
            sink.to(torch.float32).unsqueeze(0) - rowmax.squeeze(-1)
        )  # [T, N]
    else:
        sink_term = torch.zeros_like(rowmax.squeeze(-1))
    denom = exp_scores.sum(dim=-1) + sink_term  # [T, N]
    attn = exp_scores / denom.unsqueeze(-1)

    out = torch.matmul(attn.to(q.dtype), v)
    return out.to(q.dtype)


def _attend_single_query(
    q_t: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    mask: Optional[torch.Tensor] = None,
    sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention for a single query token.

    Args:
        q_t: ``[1, N, D]``.
        k: ``[S, D]``.
        v: ``[S, D]``.
        mask: ``[1, S]`` or ``[S]`` boolean.
        sink: ``[N]`` learnable sink logit.

    Returns:
        ``[1, N, D]``.
    """
    if mask is not None and mask.dim() == 1:
        mask = mask.unsqueeze(0)
    return _flash_attention_with_sink(q_t, k, v, scale, mask, sink)


# ---------------------------------------------------------------------------
# Public reference entry point
# ---------------------------------------------------------------------------

def kv_quant_sparse_attn_sharedkv_pytorch(
    q: torch.Tensor,
    kv_quant_mode: int = 1,
    ori_kv: Optional[torch.Tensor] = None,
    cmp_kv: Optional[torch.Tensor] = None,
    ori_sparse_indices: Optional[torch.Tensor] = None,
    cmp_sparse_indices: Optional[torch.Tensor] = None,
    ori_block_table: Optional[torch.Tensor] = None,
    cmp_block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_ori_kv: Optional[torch.Tensor] = None,
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    tile_size: int = 0,
    rope_head_dim: int = 0,
    softmax_scale: float = 0.0,
    cmp_ratio: int = 0,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "BSND",
    layout_kv: str = "PA_ND",
    return_softmax_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for ``kv_quant_sparse_attn_sharedkv``.

    The signature mirrors the NPU op **exactly** — same parameter names, order,
    and defaults as the schema registered in ``csrc/torch_binding.cpp`` (the
    ``npu_kv_quant_sparse_attn_sharedkv`` ``ops.def(...)``). Callers can therefore
    pass arguments positionally or by keyword exactly as they would call the real
    ``torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv``.

    Args / attribute semantics (matching the op README and host check):

    * ``q``: Query, ``[T, N1, 512]`` (TND) or ``[B, S1, N1, 512]`` (BSND), bf16.
    * ``kv_quant_mode``: KV nope quant mode. Op-fixed at ``1`` (per-tile fp8_e4m3).
    * ``ori_kv``: Sliding-window KV cache, ``[num_blocks, block_size, 1, 640]``.
      A5 dtype is ``float8_e4m3fn`` (allocated by ``AscendDeepseekV4SWACache``,
      ``cached_head_size = 512 + 128 = 640``); non-A5 is ``int8``. Both are 1-byte
      = the 640B fp8-packed layout. We accept ``uint8`` here and ``view`` to fp8;
      bf16 is never passed in (rejected by the op's host check).
    * ``cmp_kv``: compressed KV cache, same layout/dtype as ``ori_kv``.
    * ``ori_sparse_indices``: reserved (README: "预留参数，当前不生效"). Accepted, unused.
    * ``cmp_sparse_indices``: per-query top-k compressed-KV *logical token indices*,
      ``[T, 1, K]`` (TND), int32. Used only when ``cmp_ratio=4``.
    * ``ori_block_table`` / ``cmp_block_table``: PageAttention page tables,
      ``[B, max_blocks]``, int32.
    * ``cu_seqlens_q``: cumulative query token counts, ``[B+1]``, int32 (TND).
    * ``cu_seqlens_ori_kv`` / ``cu_seqlens_cmp_kv``: reserved, accepted, unused.
    * ``seqused_q``: reserved, accepted, unused.
    * ``seqused_kv``: per-batch valid ori_kv length, ``[B]``, int32.
    * ``sinks``: learnable sink logit, ``[N1]``, float32.
    * ``metadata``: AICPU core-split result (shape ``[1024]``); not simulated here.
    * ``tile_size``: nope dequant tile. Op-fixed at ``64`` (0 -> 64).
    * ``rope_head_dim``: RoPE head dim. Op-fixed at ``64`` (0 -> 64).
    * ``softmax_scale``: attention scale applied to ``Q @ K^T``.
    * ``cmp_ratio``: compression ratio, ``1`` (SWA) / ``4`` (sparse) / ``128`` (dense).
    * ``ori_mask_mode``: op-fixed ``4`` (band/SWA).
    * ``cmp_mask_mode``: op-fixed ``3`` (right-down causal).
    * ``ori_win_left`` / ``ori_win_right``: op-fixed ``127`` / ``0``.
    * ``layout_q``: ``"TND"`` (reference supports) or ``"BSND"``.
    * ``layout_kv``: op-fixed ``"PA_ND"``.
    * ``return_softmax_lse``: if True, also return a (placeholder) softmax_lse.

    Returns:
        ``(attention_out, softmax_lse)`` matching the op's 2-tuple output.
        ``attention_out``: ``[T, N1, 512]`` (TND) or ``[B, S1, N1, 512]`` (BSND),
        bf16. ``softmax_lse`` is an empty tensor unless ``return_softmax_lse`` is
        True (matching the op's InferShape: False -> shape ``[0]``); when True the
        reference returns a best-effort lse that is NOT numerically validated.
    """
    # ---- Attribute validation (mirrors host check, op-fixed values only). ----
    if kv_quant_mode != 1:
        raise ValueError(f"kv_quant_mode only support 1, got {kv_quant_mode}")
    if tile_size not in (0, 64):
        raise ValueError(f"tile_size only support 64 (or 0 for default), got {tile_size}")
    if rope_head_dim not in (0, 64):
        raise ValueError(f"rope_head_dim only support 64 (or 0 for default), got {rope_head_dim}")
    if ori_mask_mode != 4:
        raise ValueError(f"ori_mask_mode only support 4, got {ori_mask_mode}")
    if cmp_mask_mode != 3:
        raise ValueError(f"cmp_mask_mode only support 3, got {cmp_mask_mode}")
    if ori_win_left != 127:
        raise ValueError(f"ori_win_left only support 127, got {ori_win_left}")
    if ori_win_right != 0:
        raise ValueError(f"ori_win_right only support 0, got {ori_win_right}")
    if layout_kv != "PA_ND":
        raise ValueError(f"layout_kv only support PA_ND, got {layout_kv}")
    if layout_q not in ("TND", "BSND"):
        raise ValueError(f"layout_q only support TND/BSND, got {layout_q}")
    if q.shape[-1] != 512:
        raise ValueError(f"q last dim must be 512, got {q.shape[-1]}")
    if ori_kv is None:
        raise ValueError("ori_kv is required")
    if ori_kv.shape[-1] != 640:
        raise ValueError(f"ori_kv last dim must be 640, got {ori_kv.shape[-1]}")
    if cmp_ratio not in (1, 4, 128):
        raise ValueError(f"cmp_ratio must be 1/4/128, got {cmp_ratio}")
    if cmp_ratio == 4 and (cmp_sparse_indices is None or cmp_block_table is None or cmp_kv is None):
        raise ValueError("cmp_ratio=4 requires cmp_kv, cmp_block_table and cmp_sparse_indices")
    if cmp_ratio == 128 and (cmp_kv is None or cmp_block_table is None):
        raise ValueError("cmp_ratio=128 requires cmp_kv and cmp_block_table")
    if ori_block_table is None or seqused_kv is None:
        raise ValueError("ori_block_table and seqused_kv are required")
    if layout_q == "TND" and cu_seqlens_q is None:
        raise ValueError("cu_seqlens_q is required when layout_q is TND")
    # Reserved / unused inputs: ori_sparse_indices, cu_seqlens_ori_kv,
    # cu_seqlens_cmp_kv, seqused_q, metadata — accepted to match the op signature.

    # BSND layout: flatten the leading batch/seq into T so the rest of the logic
    # (written for TND) is reused unchanged.
    bsnd = layout_q == "BSND"
    if bsnd:
        B_in, S1_in, N_q = q.shape[0], q.shape[1], q.shape[2]
        q_flat = q.reshape(B_in * S1_in, N_q, q.shape[-1])
        cu_seqlens_q = (torch.arange(B_in + 1, device=q.device) * S1_in).to(torch.int32)
        q = q_flat
    else:
        N_q = q.shape[1]

    B = seqused_kv.shape[0]
    outputs = []

    for b in range(B):
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        q_b = q[q_start:q_end]
        T_b = q_end - q_start
        # Query token positions within this batch (0-indexed along the query axis).
        q_pos_b = torch.arange(T_b, device=q.device)
        seq_len = int(seqused_kv[b].item())  # valid ori_kv length for this batch

        # ------------------------------------------------------------------
        # Query <-> KV position alignment (see kernel GetSingleCoreParam +
        # ComputeS2LoopInfo). nextTokensPerBatch = S2 - S1 is the offset between
        # the query axis and the KV axis: query token ``i`` is anchored at KV
        # position ``i + nextTokensPerBatch``. This matters whenever S2 != S1
        # (prefix KV in prefill, or decode where S1=1, S2=seq_len). For pure
        # prefill with S2 == S1 the offset is 0 and the band reduces to [i-127, i].
        # ------------------------------------------------------------------
        next_tokens = seq_len - T_b
        # KV-axis anchor of each query token.
        q_kv_pos = q_pos_b + next_tokens  # [T_b]

        # ------------------------------------------------------------------
        # 1. ori_kv: gather + dequant (full valid sequence).
        # ------------------------------------------------------------------
        ori_positions = torch.arange(seq_len, device=q.device)
        ori_kv_packed = gather_page_attention_kv(
            ori_kv, ori_block_table[b : b + 1], ori_positions
        ).squeeze(1)  # [S, 640]
        ori_kv_dequant = dequant_packed_kv(ori_kv_packed)  # [S, 512]
        k_ori = ori_kv_dequant
        v_ori = ori_kv_dequant  # shared KV

        # Band (sliding-window) mask over ori_kv, anchored at the KV axis:
        # token i attends to ori_kv j iff q_kv_pos[i] - 127 <= j <= q_kv_pos[i].
        mask_ori = build_band_mask(
            q_kv_pos, ori_positions, ori_win_left, ori_win_right
        )

        # ------------------------------------------------------------------
        # 2. cmp_kv handling by mode.
        # ------------------------------------------------------------------
        if cmp_ratio == 4:
            # Sparse compressed attention: each query token has its own top-k
            # set of compressed-KV logical token indices. Filter by the causal
            # bound from the KV-axis anchor: cmp position j participates iff
            # j < (q_kv_pos[i] + 1) // cmp_ratio (kernel CopyInKvSparse s2IdLimit,
            # which mirrors s2CmpLineEndIdx = s2LineEndIdx // cmpRatio with the
            # exclusive +1 from s1RealSize). Equivalent inclusive bound:
            # j <= (q_kv_pos[i] + 1) // cmp_ratio - 1.
            cmp_pos_limit = (q_kv_pos + 1) // cmp_ratio - 1  # [T_b], inclusive
            # cmp_sparse_indices: [T, 1, K] -> per-batch [T_b, K].
            indices_b = cmp_sparse_indices[q_start:q_end, 0, :]  # [T_b, K]

            out_b_chunks = []
            for t in range(T_b):
                idx_t = indices_b[t]  # [K]
                limit = int(cmp_pos_limit[t].item())
                # Keep indices inside [0, limit] (causal bound). The kernel reads
                # indices position-by-position and stops at the first out-of-range
                # (-1 sentinel); the indexer already packs valid indices first, so
                # a boolean filter on [0, limit] reproduces the surviving set.
                valid = (idx_t >= 0) & (idx_t <= limit)
                idx_valid = idx_t.clamp(min=0)
                if valid.any():
                    cmp_kv_packed = gather_page_attention_kv(
                        cmp_kv, cmp_block_table[b : b + 1], idx_valid
                    ).squeeze(1)  # [K, 640]
                    cmp_kv_dequant = dequant_packed_kv(cmp_kv_packed)  # [K, 512]
                else:
                    cmp_kv_dequant = ori_kv_dequant.new_zeros((0, 512))

                # Sliding-window slice of ori_kv for this single query token.
                mask_ori_t = mask_ori[t]  # [S]
                k_ori_t = k_ori[~mask_ori_t]
                v_ori_t = v_ori[~mask_ori_t]

                if cmp_kv_dequant.shape[0] > 0:
                    eff_valid = valid.to(torch.bool)
                    k_cmp_t = cmp_kv_dequant[eff_valid]
                    v_cmp_t = cmp_kv_dequant[eff_valid]
                    k_t = torch.cat([k_ori_t, k_cmp_t], dim=0)
                    v_t = torch.cat([v_ori_t, v_cmp_t], dim=0)
                else:
                    k_t = k_ori_t
                    v_t = v_ori_t

                out_t = _attend_single_query(
                    q_b[t : t + 1], k_t, v_t, softmax_scale, sink=sinks
                )
                out_b_chunks.append(out_t)
            out_b = torch.cat(out_b_chunks, dim=0)

        else:
            # cmp_ratio in {1, 128}: ori_kv is shared across all query tokens,
            # optionally extended with the full compressed KV (right-down causal).
            k_parts = [k_ori]
            v_parts = [v_ori]
            cmp_positions = None

            if cmp_ratio == 128:
                cmp_len = seq_len // cmp_ratio
                if cmp_len > 0:
                    cmp_pos = torch.arange(cmp_len, device=q.device)
                    cmp_kv_packed = gather_page_attention_kv(
                        cmp_kv, cmp_block_table[b : b + 1], cmp_pos
                    ).squeeze(1)  # [cmp_len, 640]
                    cmp_kv_dequant = dequant_packed_kv(cmp_kv_packed)
                    k_parts.append(cmp_kv_dequant)
                    v_parts.append(cmp_kv_dequant)
                    cmp_positions = cmp_pos

            k_all = torch.cat(k_parts, dim=0)  # [S (+cmp_len), 512]
            v_all = torch.cat(v_parts, dim=0)
            kv_pos_all = torch.cat(
                [ori_positions]
                + ([cmp_positions] if cmp_positions is not None else []),
                dim=0,
            )

            # Band mask over ori_kv (KV-axis anchored) + right-down causal over
            # the compressed tail (j < (q_kv_pos+1)//cmp_ratio, exclusive — see §3.5).
            mask = build_band_mask(q_kv_pos, kv_pos_all, ori_win_left, ori_win_right)
            if cmp_positions is not None:
                mask_cmp = build_right_down_causal_mask(
                    q_kv_pos, cmp_positions, cmp_ratio
                )
                mask[:, -cmp_positions.shape[0] :] = mask_cmp

            out_b = _flash_attention_with_sink(
                q_b, k_all, v_all, softmax_scale, mask, sinks
            )

        outputs.append(out_b)

    attn_out = torch.cat(outputs, dim=0)
    if bsnd:
        attn_out = attn_out.reshape(B_in, S1_in, N_q, q.shape[-1])

    # Match the op's 2-tuple output. softmax_lse is empty unless requested
    # (op InferShape: return_softmax_lse=False -> shape [0]); the reference does
    # not compute a numerically-validated lse, so it is a placeholder when True.
    if return_softmax_lse:
        softmax_lse = attn_out.new_zeros((N_q, attn_out.shape[0] if not bsnd else B_in * S1_in))
    else:
        softmax_lse = attn_out.new_zeros((0,))
    return attn_out, softmax_lse


# ---------------------------------------------------------------------------
# Self-test (CPU-friendly).
# ---------------------------------------------------------------------------

def pack_random_kv(num_tokens: int, seed: int = 0) -> torch.Tensor:
    """Build a *legal* 640B packed KV buffer for tests.

    Random uint8 bytes reinterpreted as fp8_e4m3fn can hit the 0xFF NaN slot, so
    we instead quantize random bf16 values into fp8_e4m3fn / fp8_e8m0 and pack
    them back, mirroring how the model actually produces this buffer. Layout:
    ``[rope(128B bf16) | nope(448B fp8_e4m3) | scale(7B fp8_e8m0) | pad(57B)]``.
    """
    g = torch.Generator().manual_seed(seed)
    rope = torch.randn(num_tokens, 64, generator=g, dtype=torch.bfloat16)
    # fp8_e4m3 max ~448; keep nope magnitudes in range to avoid Inf.
    nope = torch.randn(num_tokens, 448, generator=g, dtype=torch.float32)
    nope_bf16 = nope.to(torch.bfloat16)
    nope_tiles = nope_bf16.view(num_tokens, 7, 64)
    # per-tile e8m0 scale = max(abs(tile)) (power-of-two rounded by the dtype).
    scale = nope_tiles.abs().amax(-1).clamp(min=1e-30)  # [num_tokens, 7]
    scale_fp8 = scale.to(torch.float8_e8m0fnu).to(torch.float32)  # [num_tokens, 7]
    # Quantize each 64-dim tile by its scale, keeping values inside fp8_e4m3 range.
    nope_q = (nope_tiles / scale_fp8.unsqueeze(-1)).clamp(-448, 448).reshape(
        num_tokens, 448
    )
    nope_fp8 = nope_q.to(torch.float8_e4m3fn).view(torch.uint8)

    buf = torch.zeros(num_tokens, 640, dtype=torch.uint8)
    buf[:, :128] = rope.view(torch.uint8)
    buf[:, 128:576] = nope_fp8
    buf[:, 576:583] = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    return buf


def _naive_sink_attention(q, k, v, scale, sink):
    """Independent reference: treat sink as a virtual KV token whose score is the
    constant ``sink[n]`` (not Q@K) and whose value is 0. Used to cross-check the
    Flash-style sink implementation above."""
    import torch.nn.functional as F

    T, N, D = q.shape
    S = k.shape[0]
    scores = (q.float() @ k.float().transpose(-2, -1)) * scale  # [T, N, S]
    # Append a virtual column with score == sink[n] (broadcast over T) and an
    # extra zero value row.
    virt = sink.to(torch.float32).view(1, N, 1).expand(T, N, 1)
    full = torch.cat([scores, virt], dim=-1)  # [T, N, S+1]
    attn = F.softmax(full, dim=-1).to(q.dtype)
    v_ext = torch.cat([v, v.new_zeros((1, D))], dim=0)  # [S+1, D], last row is 0
    return (attn @ v_ext).to(q.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- Case A: prefill-style, S2 == S1 per batch (next_tokens == 0). ----
    B = 2
    T = 16
    N_q = 4
    block_size = 16
    num_ori_blocks = 4
    num_cmp_blocks = 2
    K = 8

    q = torch.randn(T, N_q, 512, dtype=torch.bfloat16)
    cu_seqlens_q = torch.tensor([0, T // 2, T], dtype=torch.int32)
    seqused_kv = torch.tensor([T // 2, T // 2], dtype=torch.int32)

    ori_kv = pack_random_kv(num_ori_blocks * block_size, seed=1).view(
        num_ori_blocks, block_size, 1, 640
    )
    ori_block_table = torch.arange(num_ori_blocks, dtype=torch.int32).view(1, -1).repeat(B, 1)
    cmp_kv = pack_random_kv(num_cmp_blocks * block_size, seed=2).view(
        num_cmp_blocks, block_size, 1, 640
    )
    cmp_block_table = torch.arange(num_cmp_blocks, dtype=torch.int32).view(1, -1).repeat(B, 1)
    cmp_sparse_indices = torch.randint(0, max(T // 2 // 4, 1), (T, 1, K), dtype=torch.int32)
    sinks = torch.full((N_q,), -2.0, dtype=torch.float32)

    for ratio in (1, 4, 128):
        out, lse = kv_quant_sparse_attn_sharedkv_pytorch(
            q=q,
            ori_kv=ori_kv,
            ori_block_table=ori_block_table,
            seqused_kv=seqused_kv,
            cu_seqlens_q=cu_seqlens_q,
            cmp_kv=cmp_kv,
            cmp_block_table=cmp_block_table,
            cmp_sparse_indices=cmp_sparse_indices,
            sinks=sinks,
            softmax_scale=1.0 / math.sqrt(512),
            cmp_ratio=ratio,
            layout_q="TND",
        )
        print(
            f"[case A prefill, cmp_ratio={ratio}] "
            f"shape={tuple(out.shape)} dtype={out.dtype} lse={tuple(lse.shape)}"
        )

    # ---- Case B: decode-style, S1=1, S2 large (next_tokens = S2-1). ----
    # Exercises the query<->KV alignment fix: the single query token must attend
    # to the *last* ori_kv window [S2-1-127, S2-1], not position 0.
    S2 = 300
    T_dec = 1
    num_ori_blocks_dec = (S2 + block_size - 1) // block_size
    q_dec = torch.randn(T_dec, N_q, 512, dtype=torch.bfloat16)
    cu_dec = torch.tensor([0, T_dec], dtype=torch.int32)
    seq_dec = torch.tensor([S2], dtype=torch.int32)
    ori_kv_dec = pack_random_kv(num_ori_blocks_dec * block_size, seed=3).view(
        num_ori_blocks_dec, block_size, 1, 640
    )
    bt_dec = torch.arange(num_ori_blocks_dec, dtype=torch.int32).view(1, -1)
    out_dec, _ = kv_quant_sparse_attn_sharedkv_pytorch(
        q=q_dec,
        ori_kv=ori_kv_dec,
        ori_block_table=bt_dec,
        seqused_kv=seq_dec,
        cu_seqlens_q=cu_dec,
        sinks=sinks,
        softmax_scale=1.0 / math.sqrt(512),
        cmp_ratio=1,
        layout_q="TND",
    )
    print(f"[case B decode, cmp_ratio=1] shape={tuple(out_dec.shape)} (S2={S2})")

    # Cross-check the decode output against an explicit SWA over the last window.
    last_kv = dequant_packed_kv(ori_kv_dec.reshape(-1, 640)[:S2])  # [S2, 512]
    lo = max(S2 - 1 - 127, 0)
    k_win = last_kv[lo:S2].float()  # [W, 512]
    # Manual attention with sink for the single query token.
    s = (q_dec[0].float() @ k_win.transpose(-2, -1)) * (1.0 / math.sqrt(512))  # [N, W]
    m = torch.maximum(s.amax(-1), sinks)  # joint max with sink
    e = torch.exp(s - m.unsqueeze(-1))
    denom = e.sum(-1) + torch.exp(sinks - m)
    ref_dec = (e / denom.unsqueeze(-1) @ k_win).to(torch.bfloat16)  # [N, 512]
    max_err = (out_dec[0].float() - ref_dec.float()).abs().max().item()
    print(f"[case B decode] max abs err vs explicit last-window SWA: {max_err:.4e}")
    assert max_err < 0.05, f"decode alignment mismatch: {max_err}"

    # ---- Case C: sink cross-check (Flash impl vs naive virtual-token impl). ----
    q_c = torch.randn(5, N_q, 512, dtype=torch.bfloat16)
    k_c = dequant_packed_kv(pack_random_kv(7, seed=4))
    v_c = k_c
    out_flash = _flash_attention_with_sink(q_c, k_c, v_c, 1.0 / math.sqrt(512), None, sinks)
    out_naive = _naive_sink_attention(q_c, k_c, v_c, 1.0 / math.sqrt(512), sinks)
    err = (out_flash.float() - out_naive.float()).abs().max().item()
    print(f"[case C sink] max abs err Flash vs naive-virtual-token: {err:.4e}")
    assert err < 0.01, f"sink semantics mismatch: {err}"

    # ---- Case D: cmp_ratio=4 causal bound (exclusive (q_kv_pos+1)//4). ----
    # Verifies the kernel's s2CmpLineEndIdx = s2LineEndIdx // cmpRatio rule
    # (s2LineEndIdx = q_kv_pos + 1, exclusive). With S2==S1 (nextTokens=0) the
    # per-query cmp exclusive end is (i+1)//4: i in {0,1,2} attend no cmp; i>=3
    # attends cmp 0; i>=7 attends cmp 0,1.
    T_d = 8
    q_d = torch.randn(T_d, N_q, 512, dtype=torch.bfloat16)
    ori_kv_d = pack_random_kv(block_size, seed=5).view(1, block_size, 1, 640)
    bt_d = torch.zeros(1, 1, dtype=torch.int32)
    cu_d = torch.tensor([0, T_d], dtype=torch.int32)
    seq_d = torch.tensor([T_d], dtype=torch.int32)
    cmp_kv_d = pack_random_kv(block_size, seed=6).view(1, block_size, 1, 640)
    cmp_bt_d = torch.zeros(1, 1, dtype=torch.int32)
    # Every query nominally selects cmp {0, 1}; the causal bound decides which survive.
    idx_d = torch.zeros(T_d, 1, 2, dtype=torch.int32)
    idx_d[:, 0, 0] = 0
    idx_d[:, 0, 1] = 1
    out_d, _ = kv_quant_sparse_attn_sharedkv_pytorch(
        q=q_d, ori_kv=ori_kv_d, ori_block_table=bt_d, seqused_kv=seq_d, cu_seqlens_q=cu_d,
        cmp_kv=cmp_kv_d, cmp_block_table=cmp_bt_d, cmp_sparse_indices=idx_d, sinks=sinks,
        softmax_scale=1.0 / math.sqrt(512), cmp_ratio=4, layout_q="TND",
    )
    ori_dq = dequant_packed_kv(ori_kv_d.reshape(-1, 640)[:T_d])
    cmp_dq = dequant_packed_kv(cmp_kv_d.reshape(-1, 640)[:2])
    sc_d = 1.0 / math.sqrt(512)
    max_err_d = 0.0
    for i in range(T_d):
        lo = max(0, i - 127)
        cmp_end = (i + 1) // 4  # exclusive
        ksel = [ori_dq[lo : i + 1]]
        for j in range(cmp_end):
            ksel.append(cmp_dq[[j]])
        k_i = torch.cat(ksel, dim=0).float()
        s = (q_d[i].float() @ k_i.T) * sc_d
        m = torch.maximum(s.amax(-1), sinks)
        e = torch.exp(s - m.unsqueeze(-1))
        den = e.sum(-1) + torch.exp(sinks - m)
        ref_i = ((e / den.unsqueeze(-1)) @ k_i).to(torch.bfloat16)
        max_err_d = max(max_err_d, (out_d[i].float() - ref_i.float()).abs().max().item())
    print(f"[case D cmp causal] max abs err vs independent (exclusive bound): {max_err_d:.4e}")
    assert max_err_d < 0.05, f"cmp causal bound mismatch: {max_err_d}"

    print("self-test passed.")
