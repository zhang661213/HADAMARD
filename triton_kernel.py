"""
HADAMARD Triton Kernel Source

融合 kernel：Hadamard 旋转 + 归一化 + 量化 → 单 kernel 完成。
支持 FP16/BF16 输入，支持 CPU/GPU。

融合kernel内部内存分配：
  1. 旋转: x @ Q^T = (x ⊙ signs) @ H_d / √d
  2. 归一化: v = ||x||_2, x_norm = x / v
  3. 量化: indices = quantize(x_norm, centroids)
     同时存储 norms 和 indices
"""

import torch
import contextlib

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except (ImportError, AttributeError):
    _TRITON_AVAILABLE = False
    triton = None
    tl = None

_no_grad = contextlib.nullcontext
if torch is not None:
    _no_grad = torch.no_grad


# =============================================================================
# CUDA C++ Kernel Source (Triton编译用)
# =============================================================================

TURBOQUANT_KERNEL_SOURCE = r'''
#include <cuda_bf16.h>
#include <cuda_fp16.h>

using bfloat16 = __nv_bfloat16;
using half = __half;

// ============================================================================
// FWHT (Walsh-Hadamard Transform)
// ============================================================================

template <typename T>
__device__ void fwht_inplace(T* data, int d) {
    for (int stride = 1; stride < d; stride <<= 1) {
        for (int block = 0; block < d; block += stride << 1) {
            for (int j = 0; j < stride; ++j) {
                int i0 = block + j;
                int i1 = i0 + stride;
                if (i1 < d) {
                    T u = data[i0];
                    T v = data[i1];
                    data[i0] = u + v;
                    data[i1] = u - v;
                }
            }
        }
    }
}

template <typename T>
__device__ T dot_product(const T* a, const T* b, int d) {
    T sum = 0;
    for (int i = 0; i < d; ++i) sum += a[i] * b[i];
    return sum;
}
'''


# =============================================================================
# Python Triton Kernels — Hadamard rotate + quantize + bit-pack (单 kernel)
# =============================================================================

if _TRITON_AVAILABLE:
    @triton.jit
    def _triton_fused_compress_kernel(
        x_ptr, signs_ptr, norms_ptr, indices_ptr, packed_ptr,
        centroids_ptr,
        batch_stride, d_stride: tl.constexpr,
        d: tl.constexpr, d_bytes: tl.constexpr, n_groups: tl.constexpr,
        bits: tl.constexpr, indices_per_byte: tl.constexpr, mask_nlevels: tl.constexpr,
        inv_sqrt_d: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        x_base = x_ptr + batch_idx * batch_stride
        packed_base = packed_ptr + batch_idx * d_bytes
        norm_base = norms_ptr + batch_idx

        offsets = tl.arange(0, d)
        x_vals = tl.load(x_base + offsets * d_stride)
        sum2 = tl.sum(x_vals * x_vals)
        norm = tl.sqrt(sum2 + 1e-8)
        inv_norm = 1.0 / norm
        tl.store(norm_base, norm)

        x_norm = x_vals * inv_norm
        signs = tl.load(signs_ptr + offsets)
        rotated = x_norm * signs

        stride = 1
        while stride < d:
            for block_start in range(0, d, stride * 2):
                for j in range(stride):
                    if block_start + j >= d or block_start + j + stride >= d:
                        break
                    i0 = block_start + j
                    i1 = i0 + stride
                    u = tl.load(rotated + i0)
                    v = tl.load(rotated + i1)
                    tl.store(rotated + i0, u + v)
                    tl.store(rotated + i1, u - v)
            stride <<= 1
        rotated = rotated * inv_sqrt_d

        indices = tl.zeros((d,), dtype=tl.int32)
        for i in range(d):
            val = tl.load(rotated + i)
            lo = 0
            hi = mask_nlevels - 1
            for _ in range(8):
                mid = (lo + hi + 1) >> 1
                c_mid = tl.load(centroids_ptr + mid)
                cond = c_mid <= val
                lo = tl.where(cond, mid, lo)
                hi = tl.where(cond, hi, mid - 1)
            tl.store(indices + i, lo)

        shift0 = bits * (indices_per_byte - 1)
        shift1 = bits * (indices_per_byte - 2) if indices_per_byte >= 2 else 0
        shift2 = bits * (indices_per_byte - 3) if indices_per_byte >= 3 else 0
        shift3 = bits * (indices_per_byte - 4) if indices_per_byte >= 4 else 0

        if indices_per_byte == 2:
            for g in range(n_groups):
                i0 = g * 2
                idx0 = tl.load(indices + i0).to(tl.int32)
                idx1 = tl.load(indices + i0 + 1).to(tl.int32)
                byte_val = (idx0 << shift0) | (idx1 << shift1)
                tl.store(packed_base + g, byte_val.to(tl.uint8))
        elif indices_per_byte == 4:
            for g in range(n_groups):
                i0 = g * 4
                byte_val = (
                    (tl.load(indices + i0 + 0).to(tl.int32) << shift0) |
                    (tl.load(indices + i0 + 1).to(tl.int32) << shift1) |
                    (tl.load(indices + i0 + 2).to(tl.int32) << shift2) |
                    (tl.load(indices + i0 + 3).to(tl.int32) << shift3)
                )
                tl.store(packed_base + g, byte_val.to(tl.uint8))
        elif indices_per_byte == 1:
            for i in range(d):
                idx = tl.load(indices + i).to(tl.int32)
                tl.store(packed_base + i, idx.to(tl.uint8))


if _TRITON_AVAILABLE:
    @triton.jit
    def _triton_fused_decompress_kernel(
        packed_ptr, signs_ptr, norms_ptr, centroids_ptr, out_ptr,
        batch_stride, d_stride: tl.constexpr,
        d: tl.constexpr, d_bytes: tl.constexpr, n_groups: tl.constexpr,
        bits: tl.constexpr, indices_per_byte: tl.constexpr, mask_nlevels: tl.constexpr,
        sqrt_d: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        packed_base = packed_ptr + batch_idx * d_bytes
        signs_base = signs_ptr
        norm_base = norms_ptr + batch_idx
        out_base = out_ptr + batch_idx * d_stride

        # Compute shift constants (same logic as compress kernel)
        _shift0 = bits * (indices_per_byte - 1)
        _shift1 = bits * (indices_per_byte - 2) if indices_per_byte >= 2 else 0
        _shift2 = bits * (indices_per_byte - 3) if indices_per_byte >= 3 else 0
        _shift3 = bits * (indices_per_byte - 4) if indices_per_byte >= 4 else 0

        indices = tl.zeros((d,), dtype=tl.int32)
        if indices_per_byte == 2:
            for g in range(n_groups):
                byte_val = tl.load(packed_base + g).to(tl.int32)
                i0 = g * 2
                tl.store(indices + i0 + 0, (byte_val >> _shift0) & (mask_nlevels - 1))
                tl.store(indices + i0 + 1, (byte_val >> _shift1) & (mask_nlevels - 1))
        elif indices_per_byte == 4:
            for g in range(n_groups):
                byte_val = tl.load(packed_base + g).to(tl.int32)
                i0 = g * 4
                tl.store(indices + i0 + 0, (byte_val >> _shift0) & (mask_nlevels - 1))
                tl.store(indices + i0 + 1, (byte_val >> _shift1) & (mask_nlevels - 1))
                tl.store(indices + i0 + 2, (byte_val >> _shift2) & (mask_nlevels - 1))
                tl.store(indices + i0 + 3, (byte_val >> _shift3) & (mask_nlevels - 1))
        elif indices_per_byte == 1:
            for i in range(d):
                idx = tl.load(packed_base + i).to(tl.int32)
                tl.store(indices + i, idx & (mask_nlevels - 1))

        values = tl.zeros((d,), dtype=tl.float32)
        for i in range(d):
            idx = tl.load(indices + i).to(tl.int32)
            val = tl.load(centroids_ptr + idx)
            tl.store(values + i, val)

        stride = 1
        while stride < d:
            for block_start in range(0, d, stride * 2):
                for j in range(stride):
                    if block_start + j >= d or block_start + j + stride >= d:
                        break
                    i0 = block_start + j
                    i1 = i0 + stride
                    u = tl.load(values + i0)
                    v = tl.load(values + i1)
                    tl.store(values + i0, u + v)
                    tl.store(values + i1, u - v)
            stride <<= 1
        values = values * (1.0 / sqrt_d)  # undo the per-level 1/sqrt(2) normalization

        signs = tl.load(signs_ptr + tl.arange(0, d))
        values = values * signs  # apply sign flip (unrotate)

        norm = tl.load(norm_base)
        out_vals = values * norm  # denormalize
        tl.store(out_base + tl.arange(0, d) * d_stride, out_vals)


if _TRITON_AVAILABLE:
    class TurboTritonKernel:
        """Triton融合压缩kernel封装"""

        def __init__(self, head_dim, bits, signs=None, centroids=None, device='cuda'):
            if not _TRITON_AVAILABLE:
                raise ImportError('triton not installed: pip install triton')
            self.head_dim = head_dim
            self.bits = bits
            self.device = device
            self.indices_per_byte = 8 // bits
            self.n_bytes = (head_dim * bits + 7) // 8
            self.n_groups = head_dim // self.indices_per_byte
            self.mask_nlevels = (1 << bits)
            self.sqrt_d = head_dim ** 0.5
            self.inv_sqrt_d = 1.0 / self.sqrt_d
            if signs is None:
                self.signs = torch.randint(0, 2, (head_dim,), device=device).float() * 2 - 1
            else:
                self.signs = signs.to(device)
            if centroids is None:
                centroids = torch.linspace(-1, 1, self.mask_nlevels, device=device)
            self.centroids = centroids.to(device)

        def compress(self, x):
            """融合压缩：rotate → normalize → quantize → bit-pack"""
            with _no_grad():
                batch = x.shape[0]
                packed = torch.empty((batch, self.n_bytes), dtype=torch.uint8, device=self.device)
                norms = torch.empty((batch,), dtype=torch.float32, device=self.device)

                def grid(batch): return (batch, self.head_dim)

                _triton_fused_compress_kernel[grid](
                    x, self.signs, norms, None, packed, self.centroids,
                    x.stride(0), 1,
                    self.head_dim, self.n_bytes, self.n_groups,
                    self.bits, self.indices_per_byte, self.mask_nlevels, self.inv_sqrt_d,
                )
                return packed, norms

        def decompress(self, packed, norms):
            """融合解压：bit-unpack → dequantize → unrotate"""
            with _no_grad():
                batch = packed.shape[0]
                out = torch.empty((batch, self.head_dim), dtype=torch.float32, device=self.device)

                def grid(batch): return (batch, self.head_dim)

                _triton_fused_decompress_kernel[grid](
                    packed, self.signs, norms, self.centroids, out,
                    out.stride(0), 1,
                    self.head_dim, self.n_bytes, self.n_groups,
                    self.bits, self.indices_per_byte, self.mask_nlevels, self.sqrt_d,
                )
                return out

        @classmethod
        def from_compressor(cls, compressor):
            """从 MSECompressor 复用 signs 和 centroids 创建 kernel"""
            return cls(
                head_dim=compressor.head_dim,
                bits=compressor.bits,
                signs=getattr(compressor.rot, 'signs', None),
                centroids=compressor.centroids,
                device=compressor.device,
            )
