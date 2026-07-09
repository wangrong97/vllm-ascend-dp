"""Pseudo-quantization helpers for DSA attention inputs.

These helpers perform a simulated quantization round-trip in bf16/fp32. They
are intended for quick algorithmic experiments only: the returned tensors are
still high-precision tensors that downstream bf16 kernels can consume without
modification. They do NOT produce real quantized packed data.

All operations are written to be graph-mode friendly (no data-dependent Python
control flow) so they can be captured by ``torch.compile`` / torchair / ACL
graph.
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from vllm.logger import logger

def _pad_to_multiple(x: torch.Tensor, dim: int, multiple: int) -> torch.Tensor:
    """Pad ``x`` along ``dim`` so its size is a multiple of ``multiple``.

    Uses zero padding and avoids Python control flow so it is safe under
    graph capture. ``multiple`` must be a compile-time constant.
    """
    size = x.size(dim)
    remainder = size % multiple
    # remainder is an int here because ``size`` is an int under eager mode and
    # a compile-time constant under graph capture. F.pad with zero total padding
    # is a no-op, so we can always call it without branching.
    pad_amount = (multiple - remainder) % multiple
    dim_pos = dim % x.dim()
    pads = [0, 0] * x.dim()
    pads[(x.dim() - 1 - dim_pos) * 2 + 1] = pad_amount
    return F.pad(x, pads)


def pseudo_quantize_fp4_per_block(
    x: torch.Tensor,
    block_size: int = 32,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Pseudo-quantize tensor to FP4 with per-block scaling.

    Args:
        x: Input tensor of shape ``[..., last_dim]``. The last dimension is
            padded to a multiple of ``block_size`` internally if needed.
        block_size: Size of each quantization block along the last dimension.
            Must be a compile-time constant for graph capture.
        scale_dtype: Dtype of the computed scale tensor. Defaults to bf16.

    Returns:
        A tensor with the same shape and dtype as ``x`` that has been through a
        simulated FP4 round-trip.
    """
    logger.info_once(f"对kv进行FP4量化")
    original_shape = x.shape
    original_last_dim = x.size(-1)

    # Pad last dim to a multiple of block_size without data-dependent branching.
    x_padded = _pad_to_multiple(x, dim=-1, multiple=block_size)
    padded_last_dim = x_padded.size(-1)
    num_blocks = padded_last_dim // block_size

    # Collapse all leading dimensions; keep padded last dim.
    x_2d = x_padded.reshape(-1, padded_last_dim)
    x_blocks = x_2d.reshape(-1, num_blocks, block_size)

    # FP4 symmetric quantization range [-8, 7].
    fp4_max = 7.0

    # Per-block scale: max(|x|) / fp4_max. Clamp avoids div-by-zero in graph.
    scale = x_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / fp4_max
    scale = scale.to(scale_dtype)

    # Round-trip in fp32 for numerical stability.
    x_blocks_fp32 = x_blocks.to(torch.float32)
    scale_fp32 = scale.to(torch.float32)
    quant = torch.clamp(torch.round(x_blocks_fp32 / scale_fp32), -fp4_max, fp4_max)
    dequant = (quant * scale_fp32).to(x.dtype)

    # Reshape back and remove padding to restore original shape.
    # Always slice to original_last_dim; when no padding was added this is a
    # no-op identity slice. Static slice indices keep the op graph-capture safe.
    dequant_padded = dequant.reshape(x_padded.shape)
    slices = tuple(slice(None) for _ in range(x.dim()))
    slices = slices[:-1] + (slice(0, original_last_dim),)
    return dequant_padded[slices].reshape(original_shape)


def _hif8_pseudo_quantize_per_tensor(
    x: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """HiFloat8 (HiF8) per-tensor pseudo quantization.

    Strictly follows the official HiF8 format (no Python branches, TorchScript
    and graph-mode compatible).

    Args:
        x: Input tensor of dtype FP32/BF16/FP16.
        scale: Optional manual per-tensor scale. If ``None``, scale is computed
            automatically from ``max(|x|)``.

    Returns:
        A tensor with the same shape and dtype as ``x`` after a HiF8 round-trip.
    """
    # HiF8 format constants.
    MAX_HIF8 = 32768.0  # 2^15
    MIN_NORMAL_EXP = -15.0
    EPS = 1e-30

    orig_dtype = x.dtype
    xf = x.to(torch.float32)

    # 1. Compute per-tensor scale if not provided.
    # ``scale is None`` is a compile-time constant in our usage (we always pass
    # it from the per-token wrapper), so this branch is graph-mode safe.
    if scale is None:
        abs_max = torch.clamp(torch.max(torch.abs(xf)), min=EPS)
        scale = abs_max / MAX_HIF8
    x_scaled = xf / (scale + EPS)

    # 2. Extract sign and absolute value.
    sign = torch.sign(x_scaled)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    abs_x = torch.abs(x_scaled)

    # 3. Global exponent calculation.
    log_x = torch.log2(abs_x + EPS)
    e_floor = torch.floor(log_x)
    is_subnormal = e_floor < MIN_NORMAL_EXP
    e_clamped = torch.clamp(e_floor, min=MIN_NORMAL_EXP, max=15.0)

    # 4. Branch-less D-band to mantissa-width mapping (official HiF8 format).
    # D4 default: exponent ±[8, 15] -> M=1
    m_bits = torch.ones_like(e_clamped)

    # D3: exponent ±[4, 7] -> M=2
    mask_d3 = ((e_clamped >= -7.0) & (e_clamped <= -4.0)) | ((e_clamped >= 4.0) & (e_clamped <= 7.0))
    m_bits = torch.where(mask_d3, 2.0, m_bits)

    # D2: exponent ±[2, 3] -> M=3
    mask_d2 = ((e_clamped >= -3.0) & (e_clamped <= -2.0)) | ((e_clamped >= 2.0) & (e_clamped <= 3.0))
    m_bits = torch.where(mask_d2, 3.0, m_bits)

    # D1: exponent ±1 (including 0) -> M=3
    mask_d1 = (e_clamped >= -1.0) & (e_clamped <= 1.0)
    m_bits = torch.where(mask_d1, 3.0, m_bits)

    # D0 subnormal: M=4
    m_bits = torch.where(is_subnormal, 4.0, m_bits)

    # 5. Mantissa normalization and round-to-nearest quantization.
    mant_raw = abs_x / torch.pow(2.0, e_clamped)
    mant_quant = torch.round(mant_raw * torch.pow(2.0, m_bits)) / torch.pow(2.0, m_bits)

    # 6. Branch-less mantissa overflow: mant >= 2.0 -> exp+1, mant/2.
    overflow_mask = (mant_quant >= 2.0).to(xf.dtype)
    e_final = e_clamped + overflow_mask
    mant_final = mant_quant / torch.pow(2.0, overflow_mask)
    e_final = torch.clamp(e_final, max=15.0)

    # 7. Distinguish subnormal vs. normal quantized values.
    # Normal: X = (1 + M / 2^m) * 2^e
    val_normal = mant_final * torch.pow(2.0, e_final)
    # Subnormal: X = M / 2^4 * 2^-15
    val_subnormal = mant_final * (2.0 ** (-15.0 - 4.0))
    x_quant_abs = torch.where(is_subnormal, val_subnormal, val_normal)
    # Zero guard.
    x_quant_abs = torch.where(abs_x < EPS, torch.zeros_like(x_quant_abs), x_quant_abs)

    # 8. Restore sign and de-scale.
    x_quant = sign * x_quant_abs
    x_dequant = x_quant * scale

    return x_dequant.to(orig_dtype)


def _quant_hif8(x: torch.Tensor) -> torch.Tensor:
    """Raw HiFloat8 quantization (no scaling).

    Maps each value to the nearest representable HiFloat8 value. The sign is
    preserved and zero stays zero. Implemented with ``torch.where`` instead of
    in-place indexed assignment so it is safe under ``torch.compile`` / ACL
    graph capture.

    Args:
        x: Input tensor of dtype FP32/BF16/FP16.

    Returns:
        A tensor with the same shape and dtype as ``x`` after raw HiF8 rounding.
    """
    x_unsigned = torch.abs(x)
    sign = torch.sign(x)

    # Add a tiny epsilon to avoid log2(0). The epsilon depends on the input
    # dtype so that very small normal FP16 values are not pulled to a smaller
    # exponent than intended.
    # if x.dtype == torch.float16:
    #     eps = 2**-14
    # else:
    #     eps = 2**-45
    eps = 2**-45
    e = torch.floor(torch.log2(x_unsigned + eps))

    abse = e.abs()

    # Mantissa width assignment based on exponent magnitude:
    # |e| <= 3 -> 3 bits, |e| <= 7 -> 2 bits, |e| <= 15 -> 1 bit, else 0 bits.
    mant_bits = torch.where(
        abse <= 3, 3.0, torch.where(abse <= 7, 2.0, torch.where(abse <= 15, 1.0, 0.0))
    )

    res = (
        torch.floor(x_unsigned * 2.0 ** (-e + mant_bits) + 0.5)
        * 2.0 ** (e - mant_bits)
        * sign
    )
    return res


def pseudo_quantize_hif8_per_tensor_fixed_scale(
    x: torch.Tensor,
    scale: float = 8.0,
) -> torch.Tensor:
    """Pseudo-quantize a tensor to HiFloat8 with a fixed per-tensor scale.

    The tensor is first divided by ``scale``, quantized to the nearest HiFloat8
    value, then multiplied back by ``scale``. This simulates a real HiF8
    per-tensor quantization/dequantization round-trip while keeping the output
    in high precision.

    Args:
        x: Input tensor of dtype FP32/BF16/FP16.
        scale: Fixed scale applied before quantization. Defaults to 8.0.

    Returns:
        A tensor with the same shape and dtype as ``x`` after the round-trip.
    """
    logger.info_once(f"对q进行hif8 per tensor量化，scale固定为{scale}")
    original_shape = x.shape
    orig_dtype = x.dtype
    xf = x.to(torch.float32)

    x_scaled = xf / scale
    x_quant = _quant_hif8(x_scaled)
    x_dequant = x_quant * scale

    return x_dequant.to(orig_dtype).reshape(original_shape)


def pseudo_quantize_hif8_per_token(
    x: torch.Tensor,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Pseudo-quantize tensor to HiF8 with per-token scaling.

    Args:
        x: Input tensor of shape ``[num_tokens, ...]``. Per-token scale is
            computed independently for each token (the first dimension).
        scale_dtype: Dtype of the computed scale tensor. Defaults to bf16.

    Returns:
        A tensor with the same shape and dtype as ``x`` that has been through a
        simulated HiF8 round-trip.
    """
    logger.info_once(f"对q进行hif8 pertensor量化")
    original_shape = x.shape
    x_2d = x.reshape(x.size(0), -1)

    # Per-token scale: max(|x|) / 2^15. Clamp avoids div-by-zero in graph.
    xf = x_2d.to(torch.float32)
    max_abs = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    scale = max_abs / 32768.0
    scale = scale.to(scale_dtype)

    return _hif8_pseudo_quantize_per_tensor(x_2d, scale=scale).reshape(original_shape)


def pseudo_quantize_qkv_for_dsa(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_block_size: int = 32,
    kv_scale_dtype: torch.dtype = torch.bfloat16,
    q_scale_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply pseudo quantization to Q and KV before DSA sparse attention.

    Args:
        q: Query tensor, expected shape ``[num_tokens, num_heads, head_dim]``.
        kv: Key/value tensor, expected shape ``[num_tokens, num_kv_heads, head_dim]``.
        kv_block_size: Block size for per-block FP4 quantization of KV.
        kv_scale_dtype: Dtype for KV per-block scales.
        q_scale_dtype: Dtype for Q per-token scales.

    Returns:
        Tuple of pseudo-quantized ``(q, kv)``.
    """
    q_quant = pseudo_quantize_hif8_per_token(q, scale_dtype=q_scale_dtype)
    kv_quant = pseudo_quantize_fp4_per_block(kv, block_size=kv_block_size, scale_dtype=kv_scale_dtype)
    return q_quant, kv_quant
