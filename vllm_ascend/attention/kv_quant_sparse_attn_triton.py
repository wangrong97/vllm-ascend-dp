# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton implementation of ``kv_quant_sparse_attn_sharedkv`` (decode path).

This is a from-scratch Triton port of the PyTorch reference in
``kv_quant_sparse_attn_sharedkv_reference.py``. It targets the **decode** path
(including MTP speculative decode where a request carries several query
tokens) and supports all three ``cmp_ratio`` modes (1 = SWA only,
4 = sparse compressed, 128 = dense compressed).

Design goals (driven by the "must be graph-capturable" requirement):

* **Single kernel, value-free control flow.** The grid is
  ``(num_q_tokens, num_q_heads)`` — every dynamic quantity (per-seq KV length,
  page-table ids, sparse indices, causal bounds) is read *inside* the kernel
  via ``tl.load`` and consumed with ``tl.where`` masks. There is no
  ``.item()``, no Python ``for b in range(B)`` over batch, no variable-length
  slicing, no ``.any()`` sync — nothing that would break ACL graph capture.
* **In-kernel FP8 dequant.** The 640-byte packed KV buffer
  ``[rope(128B bf16) | nope(448B fp8_e4m3) | scale(7B fp8_e8m0) | pad]`` is
  dequantized on the fly: the contiguous packed buffer is reinterpreted as
  three same-storage views (``float8_e4m3fn`` / ``bfloat16`` / ``uint8``) and
  the kernel loads each part with the right dtype. ``fp8_e4m3 -> fp32`` uses
  Triton's native ``float8e4nv`` load+cast; the per-tile ``fp8_e8m0`` scale is
  decoded with the ``uint8 << 23`` bit trick (verified exact on Ascend).
* **Logical parity with the reference.** SWA band mask
  ``[q_kv_pos-127, q_kv_pos]``, right-down causal cmp bound
  ``j < (q_kv_pos+1)//cmp_ratio``, query<->KV alignment
  ``q_kv_pos = q_pos + (kv_len - tokens_per_seq)``, shared KV (K==V),
  learnable sink seeded Flash max (``m0=sink, s0=1``), and the A8C4 HiF8
  pseudo-quantization of ``exp_scores`` (fixed scale 8.0) are all reproduced.

Numerical note: the reference applies HiF8 to the *fully-materialized*
``exp(score - rowmax)`` tensor, whereas this kernel applies HiF8 per inner
tile inside online softmax (HiF8 is elementwise with a *fixed* scale, so it
commutes with the running-max rescale only up to HiF8 quantization error).
For decode (small S) the two agree to within HiF8 precision.

Inputs mirror the reference / NPU op so callers can switch between
``attn_op``, the pytorch reference, and this triton kernel with minimal glue.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Packed-KV layout constants (kernel-fixed, see reference docstring).
NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM  # 512
SCALE_DIM = NOPE_DIM // 64  # 7 per-tile e8m0 scales
PACKED_BYTES = 640
ROPE_BYTES = ROPE_DIM * 2  # 128
NOPE_BYTES = NOPE_DIM  # 448
SCALE_BYTES = SCALE_DIM  # 7
SCALE_OFFSET = ROPE_BYTES + NOPE_BYTES  # 576


@triton.jit
def _hif8_quant(x, scale):
    """HiFloat8 round-trip with a *fixed* per-tensor scale (elementwise).

    Triton port of ``pseudo_quantize_hif8_per_tensor_fixed_scale`` /
    ``_quant_hif8`` in ``pseudo_quant.py``: divide by ``scale``, round each
    value to the nearest representable HiF8 code (mantissa width selected by
    exponent magnitude), multiply back by ``scale``. Branch-less, graph-safe.
    """
    x_scaled = x / scale
    x_unsigned = tl.abs(x_scaled)
    sign = tl.where(x_scaled > 0.0, 1.0, tl.where(x_scaled < 0.0, -1.0, 0.0))
    eps = 2.0**-45
    e = tl.floor(tl.log2(x_unsigned + eps))
    abse = tl.abs(e)
    # |e|<=3 -> 3 mantissa bits, |e|<=7 -> 2, |e|<=15 -> 1, else 0.
    mant_bits = tl.where(
        abse <= 3.0,
        3.0,
        tl.where(abse <= 7.0, 2.0, tl.where(abse <= 15.0, 1.0, 0.0)),
    )
    res = tl.floor(x_unsigned * tl.exp2(-e + mant_bits) + 0.5) * tl.exp2(e - mant_bits) * sign
    return res * scale  # TEMP: hif8 disabled to bisect vs ascendc (should be `res * scale`)


@triton.jit
def _e8m0_decode(scale_fp8):
    """Decode fp8_e8m0 scale bytes to fp32.

    The bytes are loaded through the fp8 dtype path (triton-ascend's uint8
    vector load returns zeros), then value = 2^(byte-127) is recovered via the
    ``(byte << 23)`` bitcast trick.
    """
    bits = scale_fp8.to(tl.uint8, bitcast=True).to(tl.int32) << 23
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _dsa_decode_kernel(
    Q_ptr,  # [num_q_tokens, H, 512] bf16
    OriNope_ptr,  # [num_blocks, block_size, 448] float8_e4m3fn
    OriRope_ptr,  # [num_blocks, block_size, 64]  bfloat16
    OriScale_ptr,  # [num_blocks, block_size, 7]   uint8
    OriBT_ptr,  # [num_seqs, max_ori_blocks] int32
    CmpNope_ptr,
    CmpRope_ptr,
    CmpScale_ptr,
    CmpBT_ptr,  # [num_seqs, max_cmp_blocks] int32
    CmpIdx_ptr,  # [num_q_tokens, TOPK] int32 (cmp_ratio=4 only)
    SequsedKV_ptr,  # [num_seqs] int32
    Sinks_ptr,  # [H] float32
    Out_ptr,  # [num_q_tokens, H, 512] bf16
    stride_q_t,
    stride_q_h,
    stride_ori_nope,
    stride_ori_rope,
    stride_ori_scale,
    stride_cmp_nope,
    stride_cmp_rope,
    stride_cmp_scale,
    stride_ori_bt,
    stride_cmp_bt,
    stride_o_t,
    stride_o_h,
    softmax_scale_val,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CMP_BLOCK_SIZE: tl.constexpr,
    TOPK: tl.constexpr,
    CMP_RATIO: tl.constexpr,
    ORI_WIN_LEFT: tl.constexpr,
    TOKENS_PER_SEQ: tl.constexpr,
    MAX_ORI_TILES: tl.constexpr,
    MAX_CMP_TILES: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HIF8_SCALE: tl.constexpr,
):
    q_tok = tl.program_id(0)
    head = tl.program_id(1)
    seq = q_tok // TOKENS_PER_SEQ
    q_pos = q_tok % TOKENS_PER_SEQ

    kv_len = tl.load(SequsedKV_ptr + seq)
    # Query<->KV alignment: query token i is anchored at KV pos q_pos + next_tokens,
    # next_tokens = kv_len - tokens_per_seq.
    q_kv_pos = q_pos + (kv_len - TOKENS_PER_SEQ)

    offs_nope = tl.arange(0, NOPE_DIM)
    offs_rope = tl.arange(0, ROPE_DIM)
    offs_bn = tl.arange(0, BLOCK_N)

    q_nope = tl.load(Q_ptr + q_tok * stride_q_t + head * stride_q_h + offs_nope).to(tl.float32)
    q_rope = tl.load(Q_ptr + q_tok * stride_q_t + head * stride_q_h + NOPE_DIM + offs_rope).to(tl.float32)

    # Online softmax seeded with the learnable sink: m0 = sink, s0 = 1.
    sink = tl.load(Sinks_ptr + head)
    m_i = tl.full([1], sink, dtype=tl.float32)
    l_i = tl.full([1], 1.0, dtype=tl.float32)
    acc_nope = tl.zeros([NOPE_DIM], dtype=tl.float32)
    acc_rope = tl.zeros([ROPE_DIM], dtype=tl.float32)

    swa_start = tl.maximum(0, q_kv_pos - ORI_WIN_LEFT)
    swa_end = q_kv_pos  # inclusive (ori_win_right = 0)

    # ------------------------------------------------------------------
    # Phase 1: sliding-window attention over original KV (ori_mask_mode=4).
    # ------------------------------------------------------------------
    for t in range(MAX_ORI_TILES):
        j = swa_start + t * BLOCK_N + offs_bn
        valid = (j >= swa_start) & (j <= swa_end) & (j < kv_len)
        logical_block = j // BLOCK_SIZE
        block_offset = j % BLOCK_SIZE
        phys_block = tl.load(
            OriBT_ptr + seq * stride_ori_bt + logical_block,
            mask=valid,
            other=0,
        )
        row = phys_block * BLOCK_SIZE + block_offset  # flat token index
        # ---- in-kernel dequant of one KV tile ----
        nope = tl.load(
            OriNope_ptr + row[:, None] * stride_ori_nope + offs_nope[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        rope = tl.load(
            OriRope_ptr + row[:, None] * stride_ori_rope + offs_rope[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        # Load scale as fp8 (same storage) to dodge the broken uint8 vector
        # load on triton-ascend, then recover the e8m0 byte via bitcast.
        scale_fp8 = tl.load(
            OriScale_ptr + row[:, None] * stride_ori_scale + tl.arange(0, SCALE_DIM)[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scale_fp32 = _e8m0_decode(scale_fp8)
        # Broadcast per-tile scale across each 64-dim tile: [N,7] -> [N,448].
        scale_per_dim = tl.reshape(
            tl.broadcast_to(scale_fp32[:, :, None], (BLOCK_N, SCALE_DIM, TILE_SIZE)),
            (BLOCK_N, NOPE_DIM),
        )
        nope_dequant = nope * scale_per_dim

        nope_score = tl.sum(q_nope[None, :] * nope_dequant, axis=1)  # [N]
        rope_score = tl.sum(q_rope[None, :] * rope, axis=1)
        score = (nope_score + rope_score) * softmax_scale_val
        score = tl.where(valid, score, -float("inf"))

        m_block = tl.max(score, axis=0)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(score - m_new)
        p = _hif8_quant(p, HIF8_SCALE)
        p = tl.where(valid, p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc_nope = acc_nope * alpha + tl.sum(p[:, None] * nope_dequant, axis=0)
        acc_rope = acc_rope * alpha + tl.sum(p[:, None] * rope, axis=0)
        m_i = m_new

    # ------------------------------------------------------------------
    # Phase 2: compressed KV (cmp_ratio in {4, 128}; 1 = skip).
    # ------------------------------------------------------------------
    cmp_excl_end = (q_kv_pos + 1) // CMP_RATIO  # exclusive upper bound

    if CMP_RATIO == 4:
        # Sparse: per-query-token top-k compressed-KV logical indices.
        for k in range(TOPK):
            idx = tl.load(CmpIdx_ptr + q_tok * TOPK + k)
            valid = (idx >= 0) & (idx < cmp_excl_end)
            logical_block = idx // CMP_BLOCK_SIZE
            block_offset = idx % CMP_BLOCK_SIZE
            phys_block = tl.load(
                CmpBT_ptr + seq * stride_cmp_bt + logical_block,
                mask=valid,
                other=0,
            )
            row = phys_block * CMP_BLOCK_SIZE + block_offset
            nope = tl.load(CmpNope_ptr + row * stride_cmp_nope + offs_nope, mask=valid, other=0.0).to(tl.float32)
            rope = tl.load(CmpRope_ptr + row * stride_cmp_rope + offs_rope, mask=valid, other=0.0).to(tl.float32)
            scale_fp8 = tl.load(
                CmpScale_ptr + row * stride_cmp_scale + tl.arange(0, SCALE_DIM),
                mask=valid,
                other=0.0,
            )
            scale_fp32 = _e8m0_decode(scale_fp8)
            scale_per_dim = tl.reshape(
                tl.broadcast_to(tl.reshape(scale_fp32, (SCALE_DIM, 1)), (SCALE_DIM, TILE_SIZE)),
                (NOPE_DIM,),
            )
            nope_dequant = nope * scale_per_dim
            nope_score = tl.sum(q_nope * nope_dequant, axis=0)
            rope_score = tl.sum(q_rope * rope, axis=0)
            score = (nope_score + rope_score) * softmax_scale_val
            score = tl.where(valid, score, -float("inf"))
            m_new = tl.maximum(m_i, score)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(score - m_new)
            p = _hif8_quant(p, HIF8_SCALE)
            p = tl.where(valid, p, 0.0)
            l_i = l_i * alpha + p
            acc_nope = acc_nope * alpha + p * nope_dequant
            acc_rope = acc_rope * alpha + p * rope
            m_i = m_new
    elif CMP_RATIO == 128:
        # Dense: full scan over compressed KV tokens [0, cmp_len).
        cmp_len = kv_len // CMP_RATIO
        for t in range(MAX_CMP_TILES):
            j = t * BLOCK_N + offs_bn
            valid = (j < cmp_len) & (j < cmp_excl_end)
            logical_block = j // CMP_BLOCK_SIZE
            block_offset = j % CMP_BLOCK_SIZE
            phys_block = tl.load(
                CmpBT_ptr + seq * stride_cmp_bt + logical_block,
                mask=valid,
                other=0,
            )
            row = phys_block * CMP_BLOCK_SIZE + block_offset
            nope = tl.load(
                CmpNope_ptr + row[:, None] * stride_cmp_nope + offs_nope[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            rope = tl.load(
                CmpRope_ptr + row[:, None] * stride_cmp_rope + offs_rope[None, :],
                mask=valid[:, None],
                other=0.0,
            ).to(tl.float32)
            scale_fp8 = tl.load(
                CmpScale_ptr + row[:, None] * stride_cmp_scale + tl.arange(0, SCALE_DIM)[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            scale_fp32 = _e8m0_decode(scale_fp8)
            scale_per_dim = tl.reshape(
                tl.broadcast_to(scale_fp32[:, :, None], (BLOCK_N, SCALE_DIM, TILE_SIZE)),
                (BLOCK_N, NOPE_DIM),
            )
            nope_dequant = nope * scale_per_dim
            nope_score = tl.sum(q_nope[None, :] * nope_dequant, axis=1)
            rope_score = tl.sum(q_rope[None, :] * rope, axis=1)
            score = (nope_score + rope_score) * softmax_scale_val
            score = tl.where(valid, score, -float("inf"))
            m_block = tl.max(score, axis=0)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(score - m_new)
            p = _hif8_quant(p, HIF8_SCALE)
            p = tl.where(valid, p, 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc_nope = acc_nope * alpha + tl.sum(p[:, None] * nope_dequant, axis=0)
            acc_rope = acc_rope * alpha + tl.sum(p[:, None] * rope, axis=0)
            m_i = m_new
    # CMP_RATIO == 1: compressed path skipped entirely.

    out_nope = (acc_nope / l_i).to(tl.bfloat16)
    out_rope = (acc_rope / l_i).to(tl.bfloat16)
    tl.store(Out_ptr + q_tok * stride_o_t + head * stride_o_h + offs_nope, out_nope)
    tl.store(
        Out_ptr + q_tok * stride_o_t + head * stride_o_h + NOPE_DIM + offs_rope,
        out_rope,
    )


def _packed_kv_views(kv_cache: torch.Tensor):
    """Return (nope_fp8, rope_bf16, scale_u8) views over a packed KV pool.

    ``kv_cache`` is ``[num_blocks, block_size, 1, 640]`` uint8 (or any 1-byte
    dtype: float8_e4m3fn / int8 — all share storage). The three views alias the
    same memory via whole-buffer ``view(dtype)`` + strided slice, so this is
    zero-copy and graph-capture-safe.
    """
    if kv_cache.dtype != torch.uint8:
        kv_cache = kv_cache.view(torch.uint8)
    nb, bs, one, packed = kv_cache.shape
    assert packed == PACKED_BYTES, f"packed KV last dim must be {PACKED_BYTES}, got {packed}"
    assert one == 1
    # Whole-buffer dtype reinterprets are contiguous; the subsequent last-dim
    # slices are strided but Triton consumes them via the strides passed below.
    fp8_view = kv_cache.view(torch.float8_e4m3fn)[:, :, 0, :]  # [nb, bs, 640] fp8
    bf16_view = kv_cache.view(torch.bfloat16)[:, :, 0, :]  # [nb, bs, 320] bf16
    # nope starts right after rope (byte offset ROPE_BYTES=128); NOPE_BYTES is
    # the nope *length* (448), not the offset — do not use it as the start.
    nope = fp8_view[..., ROPE_BYTES : ROPE_BYTES + NOPE_DIM]  # [nb, bs, 448] fp8
    rope = bf16_view[..., :ROPE_DIM]  # [nb, bs, 64] bf16
    # Scale stored as fp8 dtype (same 1-byte storage) so the kernel can load
    # it via the working fp8 path and recover the e8m0 byte by register
    # bitcast — triton-ascend's uint8 vector load returns zeros.
    scale = fp8_view[..., SCALE_OFFSET : SCALE_OFFSET + SCALE_DIM]  # [nb, bs, 7] fp8
    return nope, rope, scale


def kv_quant_sparse_attn_triton_decode(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    ori_block_table: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    softmax_scale: float,
    cmp_ratio: int = 1,
    cmp_kv: torch.Tensor | None = None,
    cmp_block_table: torch.Tensor | None = None,
    cmp_sparse_indices: torch.Tensor | None = None,
    ori_block_size: int = 128,
    cmp_block_size: int = 128,
    ori_window_size: int = 128,
    tokens_per_seq: int = 1,
    max_cmp_tokens: int = 0,
    block_n: int = 16,
) -> torch.Tensor:
    """Graph-capturable Triton decode attention (mirrors the reference op).

    Args mirror the NPU op / pytorch reference (decode subset). All shapes are
    static at capture time; per-seq lengths and sparse indices are device
    tensors consumed inside the kernel — no host sync, so the whole call is
    safe under ACL graph capture.

    Args:
        q: ``[num_q_tokens, H, 512]`` bf16. ``num_q_tokens = num_seqs *
            tokens_per_seq`` (MTP carries ``tokens_per_seq > 1``).
        ori_kv: SWA KV pool ``[num_blocks, block_size, 1, 640]`` (1-byte dtype).
        ori_block_table: ``[num_seqs, max_ori_blocks]`` int32.
        seqused_kv: ``[num_seqs]`` int32, per-seq valid ori_kv length.
        sinks: ``[H]`` float32 learnable sink logits.
        softmax_scale: attention scale applied to ``Q @ K^T``.
        cmp_ratio: 1 (SWA only) / 4 (sparse) / 128 (dense compressed).
        cmp_kv: compressed KV pool (same layout as ori_kv), required for 4/128.
        cmp_block_table: ``[num_seqs, max_cmp_blocks]`` int32, required for 4/128.
        cmp_sparse_indices: ``[num_q_tokens, TOPK]`` int32, required for 4.
        ori_block_size / cmp_block_size: page block sizes (tokens).
        ori_window_size: SWA window (``ori_win_left + 1``, default 128).
        tokens_per_seq: query tokens per request (1 = plain decode, >1 = MTP).
        max_cmp_tokens: static upper bound on ``kv_len // cmp_ratio`` for
            ratio=128 (drives the compile-time scan range). Ignored otherwise.
        block_n: KV tile size loaded per inner iteration.

    Returns:
        ``[num_q_tokens, H, 512]`` bf16 attention output.
    """
    num_q_tokens, num_heads, head_dim = q.shape
    assert head_dim == HEAD_DIM, f"head_dim must be {HEAD_DIM}, got {head_dim}"
    num_seqs = seqused_kv.shape[0]
    assert num_q_tokens == num_seqs * tokens_per_seq, (
        f"num_q_tokens ({num_q_tokens}) must equal num_seqs ({num_seqs}) * tokens_per_seq ({tokens_per_seq})"
    )
    assert q.dtype == torch.bfloat16
    assert cmp_ratio in (1, 4, 128), f"cmp_ratio must be 1/4/128, got {cmp_ratio}"

    ori_nope, ori_rope, ori_scale = _packed_kv_views(ori_kv)

    if cmp_ratio == 1:
        # cmp tensors unused but kernel needs valid pointers; pass ori's.
        cmp_nope, cmp_rope, cmp_scale = ori_nope, ori_rope, ori_scale
        cmp_bt = ori_block_table
        cmp_sparse_indices = q.new_full((num_q_tokens, 1), -1, dtype=torch.int32)
        topk = 1
        max_cmp_tiles = 1
    else:
        assert cmp_kv is not None and cmp_block_table is not None
        cmp_nope, cmp_rope, cmp_scale = _packed_kv_views(cmp_kv)
        cmp_bt = cmp_block_table
        if cmp_ratio == 4:
            assert cmp_sparse_indices is not None
            topk = cmp_sparse_indices.shape[1]
            max_cmp_tiles = 1
        else:  # 128
            cmp_sparse_indices = q.new_full((num_q_tokens, 1), -1, dtype=torch.int32)
            topk = 1
            mt = max(max_cmp_tokens, 1)
            max_cmp_tiles = (mt + block_n - 1) // block_n

    win_left = ori_window_size - 1  # ori_win_left
    max_ori_tiles = (ori_window_size + block_n - 1) // block_n

    out = torch.empty_like(q)
    grid = (num_q_tokens, num_heads)

    _dsa_decode_kernel[grid](
        q,
        ori_nope,
        ori_rope,
        ori_scale,
        ori_block_table,
        cmp_nope,
        cmp_rope,
        cmp_scale,
        cmp_bt,
        cmp_sparse_indices,
        seqused_kv,
        sinks,
        out,
        q.stride(0),
        q.stride(1),
        ori_nope.stride(1),  # token-in-block stride (640 fp8 elements)
        ori_rope.stride(1),  # 320 bf16 elements
        ori_scale.stride(1),  # 640 u8 elements
        cmp_nope.stride(1),
        cmp_rope.stride(1),
        cmp_scale.stride(1),
        ori_block_table.stride(0),
        cmp_bt.stride(0),
        out.stride(0),
        out.stride(1),
        softmax_scale,
        HEAD_DIM=HEAD_DIM,
        NOPE_DIM=NOPE_DIM,
        ROPE_DIM=ROPE_DIM,
        SCALE_DIM=SCALE_DIM,
        TILE_SIZE=64,
        BLOCK_SIZE=ori_block_size,
        CMP_BLOCK_SIZE=cmp_block_size,
        TOPK=topk,
        CMP_RATIO=cmp_ratio,
        ORI_WIN_LEFT=win_left,
        TOKENS_PER_SEQ=tokens_per_seq,
        MAX_ORI_TILES=max_ori_tiles,
        MAX_CMP_TILES=max_cmp_tiles,
        BLOCK_N=block_n,
        HIF8_SCALE=8.0,
    )
    return out
