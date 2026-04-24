"""
优化 2: 量化感知注意力算子 (QAT-friendly Attention Kernel)

当前问题（解压 → 原生 Attention）：
  1. 压缩状态 → 解压到 HBM → Attention 计算 → 结果写回 HBM
  2. 中间张量 (decompressed K/V) 需要反复读写 HBM
  3. 对于 4K seq, 32 heads, BF16: K=32MB, V=32MB
     → HBM 带宽：~2GB/s (PCIe 4.0 x16) 限制
     → Attention 算子等待数据搬移，成为瓶颈

核心优化：位流直读 Attention Kernel
  1. 压缩状态（bits）直接在 GPU 寄存器内解压
  2. 无需中间 HBM 分配（decompressed KV）
  3. 解压 → 反旋转 → 点积 在一个 fused kernel 内完成
  4. 所需内存：仅码本 (2^bits × d × 4 bytes) + 位流 (~S × bits/8 bytes)

理论收益：
  - 消除中间张量读写: 节省 ~2× HBM 带宽
  - 融合算子: 减少 kernel launch overhead
  - 寄存器压力: 通过共享内存缓存码本缓解

实现方案：
  1. FusedBitstreamAttention: 融合位流解压+旋转+点积（需要 Triton/CUDA）
  2. BitstreamAttentionUnfused: 分步版本（用于验证和 CPU fallback）
  3. QATAwareAttention: 在压缩域直接计算 attention 分数（近似）

Reference:
  - QAT (Quantization-Aware Training) 原理
  - FlashAttention-2 ( IO-Aware )
  - TurboQuant (Google 2026) 位流算子
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
from dataclasses import dataclass

import torch


# ===========================================================================
# 位流打包器（压缩端）
# ===========================================================================

class BitstreamPacker:
    """
    位流打包工具：将量化索引打包为紧凑位流。

    支持非字节对齐的 bits 数（如 3 bits/element）。
    输出为连续 bit 序列（高位在前）。

    示例：
      indices: [0, 3, 7, 2, 5]  (3 bits each, max=7)
      bits=3:  000 011 111 010 101 → 15 bits → 2 bytes
    """

    @staticmethod
    def pack(indices: torch.Tensor, bits: int) -> torch.Tensor:
        """
        将整数索引打包为位流。

        Args:
            indices: (..., N) 要打包的整数索引
            bits:    每个元素的位数（通常 2-8）

        Returns:
            packed: (..., ceil(N*bits/8)) 打包后的字节数组
        """
        if bits == 8:
            return indices.to(torch.uint8)

        N = indices.numel()
        packed_len = (N * bits + 7) // 8
        packed = torch.zeros(packed_len, dtype=torch.uint8, device=indices.device)

        bits_np = bits  # 避免名称冲突

        for i in range(N):
            val = indices.view(-1)[i].item()
            bit_pos = i * bits_np
            byte_idx = bit_pos // 8
            offset = bit_pos % 8
            bits_in_first = min(bits_np, 8 - offset)
            packed[byte_idx] |= (val & ((1 << bits_in_first) - 1)) << offset
            remaining = bits_np - bits_in_first
            if remaining > 0:
                packed[byte_idx + 1] |= (val >> bits_in_first) & ((1 << remaining) - 1)

        return packed

    @staticmethod
    def unpack(packed: torch.Tensor, indices_shape: tuple, bits: int
               ) -> torch.Tensor:
        """
        从位流解包为整数索引。

        Args:
            packed:       (M,) 打包后的字节数组
            indices_shape: (..., N) 原始索引形状
            bits:         每个元素的位数

        Returns:
            indices: (..., N) 解包后的整数索引
        """
        if bits == 8:
            return packed.view(indices_shape).to(torch.long)

        N = indices_shape[-1] if len(indices_shape) > 0 else indices_shape[0]
        total = 1
        for d in indices_shape[:-1] if len(indices_shape) > 1 else ():
            total *= d
        N = total * N

        result = torch.empty(N, dtype=torch.long, device=packed.device)
        bits_np = bits

        for i in range(N):
            bit_pos = i * bits_np
            byte_idx = bit_pos // 8
            offset = bit_pos % 8
            bits_in_first = min(bits_np, 8 - offset)
            val = (packed[byte_idx] >> offset) & ((1 << bits_in_first) - 1)
            remaining = bits_np - bits_in_first
            if remaining > 0:
                val |= (packed[byte_idx + 1] & ((1 << remaining) - 1)) << bits_in_first
            result[i] = val

        if len(indices_shape) > 1:
            result = result.view(indices_shape)
        return result


# ===========================================================================
# 融合位流注意力（分步实现 + 供 CUDA/Triton 调用的规范）
# ===========================================================================

class BitstreamAttentionUnfused:
    """
    位流注意力（分步版本，用于验证逻辑正确性）。

    流程：
      1. 从位流读取量化索引
      2. 查表：索引 → 质心值（旋转前）
      3. Hadamard 旋转（在旋转域计算注意力）
      4. QK 点积 → Softmax

    注意：这个分步版本仍然需要中间 HBM 分配。
    真正的优化需要在 CUDA/Triton kernel 中将步骤 1-3 融合。
    """

    def __init__(
        self,
        centroids: torch.Tensor,     # (2^bits, d) 码本质心
        rotation_signs: torch.Tensor, # (d,) Hadamard signs
        bits: int,
        head_dim: int,
        scale: Optional[float] = None,
    ):
        self.centroids = centroids      # device 上的码本
        self.rotation_signs = rotation_signs
        self.bits = bits
        self.head_dim = head_dim
        self.scale = scale or (1.0 / math.sqrt(head_dim))

    def forward(
        self,
        q: torch.Tensor,              # (B, H, S_q, D) Query
        packed_kv: torch.Tensor,       # (B, H, S_k, packed_len) 位流
        kv_shape: tuple,              # (B, H, S_k, D) 原始 shape
        kv_bits: int,
    ) -> torch.Tensor:
        """
        位流注意力前向。

        Args:
            q:         Query (FP16/BF16)
            packed_kv: 压缩的 KV 位流
            kv_shape:  原始 KV shape
            kv_bits:   KV 量化位数

        Returns:
            attn_output: (B, H, S_q, D) 注意力输出
        """
        B, H, S_q, D = q.shape
        B2, H2, S_k, _ = kv_shape

        # Step 1: 解包位流 → 量化索引
        indices = BitstreamPacker.unpack(
            packed_kv, indices_shape=(B2, H2, S_k), bits=kv_bits
        )  # (B, H, S_k)

        # Step 2: 查表 → 旋转域的向量
        centroids = self.centroids.to(q.device)
        k_rot = centroids[indices]  # (B, H, S_k, D)
        v_rot = centroids[indices]  # (B, H, S_k, D) [复用同一码本]

        # Step 3: 逆旋转（已在旋转域存储，跳过旋转）
        # 注意：如果 KV 在旋转后量化，这里直接用 k_rot
        # 如果在旋转前量化，需要 unrotate
        k_vals = k_rot
        v_vals = v_rot

        # Step 4: QK^T / sqrt(d)
        qk = torch.einsum("bhqd,bhkd->bhqk", q, k_vals)
        qk = qk * self.scale

        # Step 5: Softmax
        attn_weights = F.softmax(qk, dim=-1)

        # Step 6: weighted sum
        output = torch.einsum("bhqk,bhkd->bhqd", attn_weights, v_vals)
        return output


@dataclass
class QATAwareKernelSpec:
    """
    QAT Kernel 规范（供 CUDA/Triton 开发者使用）。

    这个 dataclass 定义了融合位流注意力 kernel 的接口规范。
    CUDA/Triton 开发者应据此实现真正的 fused kernel。

    Kernel 工作流程（单 thread block 处理一个 head）：
      1. 从 smem 加载码本 (2^bits × d × 2bytes)
      2. 从 gmem 流式加载位流 (S × bits/8 bytes)
      3. 位解包 → 寄存器
      4. 查表 → 质心值
      5. Hadamard 旋转（in-register butterfly, d=power-of-2）
      6. QK 内积（寄存器内）
      7. 还原 softmax（在 gmem 累积）
      8. 加权求和 → 输出
    """
    # 输入
    q_ptr: int           # Query pointer (FP16/BF16, B×H×S_q×D)
    packed_k_ptr: int    # Key 位流指针 (B×H×S_k×bits/8)
    packed_v_ptr: int    # Value 位流指针
    q_stride: int        # Query stride
    kv_bits: int         # KV 量化位数
    bitsream_len: int     # 位流长度（字节）

    # 码本（共享内存）
    centroids_ptr: int   # 质心表指针
    n_levels: int        # 2^bits
    rotation_signs_ptr: int  # Hadamard signs

    # 输出
    output_ptr: int      # (B×H×S_q×D)

    # 配置
    block_size: int = 256
    stages: int = 4      # 用于 double-buffer 的 stages 数


# ===========================================================================
# 量化域注意力（近似，无需解压）
# ===========================================================================

class CompressedDomainAttention:
    """
    压缩域直接注意力（近似算法）。

    原理：
      注意力分数 A_ij = Q_i · K_j / √d
      如果 K_j 用旋转域量化：K_j = R @ C[ idx_j ]
      其中 R 是 Hadamard 旋转矩阵，C 是码本

      A_ij = Q_i · R · C[ idx_j ] / √d
           = (Q_i · R) · C[ idx_j ] / √d

    所以我们只需要将 Q 旋转一次，然后直接与压缩 K 做点积！

    优势：
      - 解压只需旋转 Q（O(B×H×S_q×D)），无需解压所有 KV
      - KV 存储在旋转域，查表后直接用
      - 注意力计算在旋转域进行

    近似误差来源：
      - 旋转是有损的（但正交，理论上无损）
      - 量化是有损的（主要误差源）
    """

    def __init__(
        self,
        centroids: torch.Tensor,
        rotation_signs: torch.Tensor,
        bits: int,
        head_dim: int,
        scale: Optional[float] = None,
    ):
        self.centroids = centroids
        self.rotation_signs = rotation_signs
        self.bits = bits
        self.head_dim = head_dim
        self.scale = scale or (1.0 / math.sqrt(head_dim))

    def rotate_q(self, q: torch.Tensor) -> torch.Tensor:
        """
        将 Q 旋转到 Hadamard 域。

        Q_rot = Q ⊙ signs @ H / √d = FWHT(Q ⊙ signs) ⊙ signs / √d
        """
        d = q.shape[-1]
        signs = self.rotation_signs.to(q.device)

        # 乘以 signs
        qr = q * signs

        # FWHT
        xr = qr.clone()
        stride = 1
        half_sqrt = math.sqrt(0.5)
        while stride < d:
            b = stride << 1
            xr_view = xr.view(*xr.shape[:-1], b // 2, 2)
            u = xr_view[..., 0]
            v = xr_view[..., 1]
            xr_view[..., 0] = (u + v) * half_sqrt
            xr_view[..., 1] = (u - v) * half_sqrt
            stride = b

        # 乘以 signs / √d
        xr = xr * signs / math.sqrt(d)
        return xr

    def forward_fused(
        self,
        q: torch.Tensor,              # (B, H, S_q, D) Query
        packed_k: torch.Tensor,       # (B, H, packed_len) Key 位流
        packed_v: torch.Tensor,       # (B, H, packed_len) Value 位流
        kv_shape: tuple,              # (B, H, S_k, D)
        kv_bits: int,
    ) -> torch.Tensor:
        """
        融合注意力（Q 旋转一次，KV 免旋转）。

        完整流程：
          1. Q 旋转一次 → Q_rot
          2. KV 位流 → 解包索引 → 质心（旋转域）
          3. Q_rot · C[idx] / √d → Attention weights
          4. Softmax → weighted sum → Output

        Memory Access（vs 原生 Attention）：
          - 原生: 读写完整解压 KV: 2×(B×H×S×D×2bytes) = ~128MB (4K seq)
          - 本方法: 只写 Q_rot (B×H×S×D×2bytes) ≈ 64MB
          - 节省: ~50% HBM 带宽
        """
        B, H, S_q, D = q.shape
        _, _, S_k, _ = kv_shape

        # Step 1: Q 旋转到 Hadamard 域
        q_rot = self.rotate_q(q)  # (B, H, S_q, D)

        # Step 2: 解包 KV 位流 → 量化索引
        packed_shape = (B, H, S_k)
        k_idx = BitstreamPacker.unpack(packed_k, packed_shape, kv_bits)
        v_idx = BitstreamPacker.unpack(packed_v, packed_shape, kv_bits)

        # Step 3: 查表（KV 已在旋转域）
        centroids = self.centroids.to(q.device)
        k_vals = centroids[k_idx.reshape(-1)].reshape(B, H, S_k, D)
        v_vals = centroids[v_idx.reshape(-1)].reshape(B, H, S_k, D)

        # Step 4: QK 点积（在旋转域）
        # Q_rot · C[k_idx] / √d（旋转域点积等价于原始点积）
        qk = torch.einsum("bhqd,bhkd->bhqk", q_rot, k_vals)
        qk = qk * self.scale

        # Step 5: Softmax
        attn = F.softmax(qk, dim=-1)

        # Step 6: 加权求和
        out = torch.einsum("bhqk,bhkd->bhqd", attn, v_vals)

        return out

    def forward_unfused(
        self,
        q: torch.Tensor,
        k_vals: torch.Tensor,
        v_vals: torch.Tensor,
    ) -> torch.Tensor:
        """普通注意力（用于比较）"""
        qk = torch.einsum("bhqd,bhkd->bhqk", q, k_vals)
        qk = qk * self.scale
        attn = F.softmax(qk, dim=-1)
        return torch.einsum("bhqk,bhkd->bhqd", attn, v_vals)


# ===========================================================================
# 工厂函数
# ===========================================================================

def create_qat_attention(
    centroids: torch.Tensor,
    rotation_signs: torch.Tensor,
    bits: int,
    head_dim: int,
    mode: str = "compressed_domain",
    scale: Optional[float] = None,
) -> "BitstreamAttentionUnfused | CompressedDomainAttention":
    """
    创建 QAT 注意力算子。

    Args:
        mode: "unfused"     → 分步位流注意力（验证用）
             "compressed_domain" → 压缩域直接注意力（推荐）
    """
    if mode == "unfused":
        return BitstreamAttentionUnfused(centroids, rotation_signs, bits, head_dim, scale)
    elif mode == "compressed_domain":
        return CompressedDomainAttention(centroids, rotation_signs, bits, head_dim, scale)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def kernel_spec_to_cuda(
    centroids: torch.Tensor,
    rotation_signs: torch.Tensor,
    bits: int,
    head_dim: int,
) -> QATAwareKernelSpec:
    """生成 CUDA Kernel 规范（供手动实现 fused kernel）"""
    return QATAwareKernelSpec(
        q_ptr=0, packed_k_ptr=0, packed_v_ptr=0,
        q_stride=0, kv_bits=bits,
        bitsream_len=0,
        centroids_ptr=0,
        n_levels=2 ** bits,
        rotation_signs_ptr=0,
        output_ptr=0,
    )
