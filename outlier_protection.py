"""
优化 4: 异常值保护机制 (Outlier Protection)

问题：
  - V3 方案依赖纯 MSE 量化，在某些 Head 激活值极度稀疏或存在极端
    Off-distribution 数据时，异常值被强制量化到最近质心
  - 高频信息（边缘、突变点）的关键细节丢失
  - 3σ 原则：正常分布下 99.7% 在 ±3σ 内，但 LLM 激活往往重尾

解决方案：混合存储
  - 正常 token：进入 TurboQuant 码本量化（高效压缩）
  - 异常值 token：直接存储原始 FP16 值 + 位置索引
  - 由于异常值极少（<0.3%），开销极小

数据结构：
  - outlier_indices: List[int]  异常值 token 的序列位置
  - outlier_values:   (B, H, N_outlier, D) 原始 FP16 值
  - compressed_kv:   BitPacker 打包的正常 token

解码时重建：
  - 从 compressed_kv 解压正常部分
  - 从 outlier_values 填入异常位置
  - 合并得到完整 KV

压缩收益分析（假设 0.3% 异常值）：
  - 正常 token: (S × 99.7%) × (kb/8) bytes
  - 异常值:     (S × 0.3%) × D × 2 bytes
  - 额外开销:   outlier_indices (S × 0.3% × 2 bytes)
  - 纯量化 baseline: S × D × 2 × (kb/16)
  - 混合 overhead:    ~1-3%（可接受，换取质量提升）

Reference: 
  - Transformer INT8 量化中的 outlier channel 分离 (LLM.int8())
  - Sparse Quantization (SPQ) 思路
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

import torch


# ===========================================================================
# 数据结构
# ===========================================================================

@dataclass
class OutlierMask:
    """
    异常值掩码。

    由 find_outliers() 生成，描述哪些 token 是异常值。
    """
    indices: torch.Tensor          # (N_outlier,) 异常值 token 位置
    n_total: int                   # 总 token 数
    sigma_threshold: float = 3.0   # 检测阈值（σ 倍数）

    @property
    def n_outlier(self) -> int:
        return len(self.indices)

    @property
    def outlier_ratio(self) -> float:
        return self.n_outlier / max(self.n_total, 1)

    @property
    def normal_mask(self) -> torch.Tensor:
        """返回 (n_total,) bool tensor，正常位置为 True"""
        m = torch.ones(self.n_total, dtype=torch.bool)
        m[self.indices] = False
        return m


@dataclass
class OutlierAwareCompressedKV:
    """
    带异常值保护的压缩 KV 状态。

    存储格式：
      1. normal_key_bits:   BitPacker 打包的正常 key 索引
      2. normal_val_bits:   BitPacker 打包的正常 value 索引
      3. outlier_kv:        (B×H×N_outlier×D) 异常值原始 FP16
      4. outlier_indices:   (N_outlier,) 异常值位置
      5. metadata:          元信息（用于解码）
    """
    # 正常 token 压缩数据
    normal_key_bits: torch.Tensor   # 位流
    normal_val_bits: torch.Tensor   # 位流
    key_bits: int
    val_bits: int
    normal_shape: Tuple[int, int, int, int]  # (B, H, S_normal, D)

    # 异常值数据
    outlier_keys: Optional[torch.Tensor] = None   # (B, H, N_outlier, D)
    outlier_values: Optional[torch.Tensor] = None  # (B, H, N_outlier, D)
    outlier_indices: Optional[torch.Tensor] = None # (N_outlier,)

    # 元信息
    sigma_threshold: float = 3.0
    n_total: int = 0
    n_normal: int = 0
    n_outlier: int = 0

    # 序列化支持
    def serialize_to_bytes(self) -> bytes:
        """
        序列化为字节流（用于网络传输 / 分布式）。

        格式：
          [4 bytes: magic = b'TQOP']
          [4 bytes: header_size]
          [header_size bytes: JSON header]
          [key_bits bytes: normal_key_bits]
          [val_bits bytes: normal_val_bits]
          [outlier_keys bytes: outlier_keys (FP16)]
          [outlier_values bytes: outlier_values (FP16)]
          [outlier_indices bytes: outlier_indices]
        """
        import json, struct

        header = {
            "key_bits": self.key_bits,
            "val_bits": self.val_bits,
            "normal_shape": list(self.normal_shape),
            "sigma_threshold": self.sigma_threshold,
            "n_total": self.n_total,
            "n_normal": self.n_normal,
            "n_outlier": self.n_outlier,
        }
        header_bytes = json.dumps(header).encode("utf-8")

        # 打包数据
        key_bits = self.normal_key_bits.detach().cpu().numpy().tobytes()
        val_bits = self.normal_val_bits.detach().cpu().numpy().tobytes()

        outlier_k = (self.outlier_keys.detach().cpu().numpy().tobytes()
                     if self.outlier_keys is not None else b"")
        outlier_v = (self.outlier_values.detach().cpu().numpy().tobytes()
                     if self.outlier_values is not None else b"")
        outlier_i = (self.outlier_indices.detach().cpu().numpy().tobytes()
                     if self.outlier_indices is not None else b"")

        magic = b"TQOP"
        header_size_bytes = struct.pack("<I", len(header_bytes))

        return (magic + header_size_bytes + header_bytes +
                key_bits + val_bits + outlier_k + outlier_v + outlier_i)

    @classmethod
    def deserialize_from_bytes(cls, data: bytes) -> "OutlierAwareCompressedKV":
        """从字节流反序列化"""
        import json, struct, numpy as np

        magic = data[:4]
        if magic != b"TQOP":
            raise ValueError(f"Invalid magic: {magic}")

        header_size = struct.unpack("<I", data[4:8])[0]
        header = json.loads(data[8:8+header_size].decode("utf-8"))

        pos = 8 + header_size

        normal_key = np.frombuffer(data[pos:pos+header["n_normal"]], dtype=np.uint8)
        pos += header["n_normal"]

        normal_val = np.frombuffer(data[pos:pos+header["n_normal"]], dtype=np.uint8)
        pos += header["n_normal"]

        def read_tensor(pos, shape, dtype):
            n_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
            return torch.from_numpy(np.frombuffer(data[pos:pos+n_bytes], dtype=dtype).reshape(shape))

        outlier_keys = None
        outlier_vals = None
        outlier_idx = None

        if header["n_outlier"] > 0:
            shape_k = (header["normal_shape"][0], header["normal_shape"][1],
                       header["n_outlier"], header["normal_shape"][3])
            outlier_keys = read_tensor(pos, shape_k, np.float16)
            pos += outlier_keys.numel() * 2
            outlier_vals = read_tensor(pos, shape_k, np.float16)
            pos += outlier_vals.numel() * 2
            outlier_idx = read_tensor(pos, (header["n_outlier"],), np.int64)

        return cls(
            normal_key_bits=normal_key,
            normal_val_bits=normal_val,
            key_bits=header["key_bits"],
            val_bits=header["val_bits"],
            normal_shape=tuple(header["normal_shape"]),
            outlier_keys=outlier_keys,
            outlier_values=outlier_vals,
            outlier_indices=outlier_idx,
            sigma_threshold=header["sigma_threshold"],
            n_total=header["n_total"],
            n_normal=header["n_normal"],
            n_outlier=header["n_outlier"],
        )


# ===========================================================================
# 异常值检测
# ===========================================================================

def find_outliers(
    x: torch.Tensor,
    sigma_threshold: float = 3.0,
    per_channel: bool = True,
) -> OutlierMask:
    """
    检测张量中的异常值。

    使用 3σ 原则：
      - 计算均值 μ 和标准差 σ
      - 异常值：|x - μ| > sigma_threshold × σ

    Args:
        x:           (B, H, S, D) 或 (B×H×S, D) 输入张量
        sigma_threshold: σ 倍数阈值（默认 3.0）
        per_channel: True=每个 head 独立计算，False=全局计算

    Returns:
        OutlierMask
    """
    orig_shape = x.shape
    flat = x.reshape(-1, x.shape[-1])  # (N, D)

    if per_channel:
        # 每列（每个维度）独立计算 μ, σ
        mu = flat.mean(dim=0)          # (D,)
        sigma = flat.std(dim=0) + 1e-8  # (D,)
    else:
        mu = flat.mean()
        sigma = flat.std() + 1e-8

    # 计算偏离度
    deviations = (flat - mu) / sigma.unsqueeze(0)  # (N, D)
    max_dev = deviations.abs().max(dim=1)[0]  # (N,) 每个 token 的最大偏离

    # 异常值位置
    outlier_bool = max_dev > sigma_threshold
    outlier_indices = torch.where(outlier_bool)[0]

    return OutlierMask(
        indices=outlier_indices,
        n_total=flat.shape[0],
        sigma_threshold=sigma_threshold,
    )


def analyze_outlier_stats(
    keys: torch.Tensor,
    values: torch.Tensor,
    sigma_threshold: float = 3.0,
) -> Dict[str, float]:
    """
    分析 KV Cache 中的异常值分布统计。

    Returns:
        dict with outlier ratios for keys/values
    """
    k_mask = find_outliers(keys, sigma_threshold=sigma_threshold)
    v_mask = find_outliers(values, sigma_threshold=sigma_threshold)

    return {
        "key_outlier_ratio": k_mask.outlier_ratio,
        "val_outlier_ratio": v_mask.outlier_ratio,
        "key_outlier_count": k_mask.n_outlier,
        "val_outlier_count": v_mask.n_outlier,
        "avg_token_outliers": (k_mask.n_outlier + v_mask.n_outlier) / 2,
    }


# ===========================================================================
# 异常值感知压缩器
# ===========================================================================

class OutlierAwareTurboQuant:
    """
    带异常值保护的 TurboQuant 压缩器。

    流程：
      1. 检测异常值 token（用 find_outliers）
      2. 分离：正常 token → TurboQuant，异常 token → 原始 FP16
      3. 存储：compressed_state + outlier_values + outlier_indices

    解压：
      1. 解压正常部分
      2. 用 outlier_values 填入异常位置
      3. 合并 → 完整 KV
    """

    def __init__(
        self,
        key_bits: int = 4,
        val_bits: int = 2,
        sigma_threshold: float = 3.0,
        device: str = "cpu",
    ):
        self.key_bits = key_bits
        self.val_bits = val_bits
        self.sigma_threshold = sigma_threshold
        self.device = device

    @torch.no_grad()
    def compress(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        centroid_k: torch.Tensor,
        centroid_v: torch.Tensor,
        key_packer,
        val_packer,
        rot_fn,
        unrot_fn,
    ) -> OutlierAwareCompressedKV:
        """
        异常值感知压缩。

        Args:
            keys/values: (B, H, S, D) 原始 KV
            centroid_k/v: 码本质心
            key_packer/val_packer: BitPacker
            rot_fn/unrot_fn: 旋转函数

        Returns:
            OutlierAwareCompressedKV
        """
        B, H, S, D = keys.shape
        flat_k = keys.reshape(-1, D)
        flat_v = values.reshape(-1, D)

        # Step 1: 检测异常值
        k_mask = find_outliers(keys, sigma_threshold=self.sigma_threshold)
        v_mask = find_outliers(values, sigma_threshold=self.sigma_threshold)

        # 异常 token = union(k, v 的异常)
        all_outlier_set = set(k_mask.indices.cpu().tolist()) | set(v_mask.indices.cpu().tolist())
        outlier_indices = sorted(all_outlier_set)
        outlier_indices_t = torch.tensor(outlier_indices, dtype=torch.long, device=keys.device)
        n_outlier = len(outlier_indices)

        # 正常 token
        normal_mask = ~torch.zeros(S, dtype=torch.bool, device=keys.device)
        normal_mask[outlier_indices_t] = False
        normal_indices = torch.where(normal_mask)[0]

        # Step 2: 旋转
        k_rot = rot_fn(keys.float()).reshape(-1, D)
        v_rot = rot_fn(values.float()).reshape(-1, D)

        # Step 3: 分离
        k_normal = k_rot[normal_indices]
        v_normal = v_rot[normal_indices]
        k_outlier = flat_k[outlier_indices_t] if n_outlier > 0 else None
        v_outlier = flat_v[outlier_indices_t] if n_outlier > 0 else None

        # Step 4: 量化正常 token
        H_dim = B * H
        S_normal = len(normal_indices)
        C_dim = S_normal

        k_dists = torch.cdist(k_normal.unsqueeze(0), centroid_k.unsqueeze(0)).squeeze(0)
        v_dists = torch.cdist(v_normal.unsqueeze(0), centroid_v.unsqueeze(0)).squeeze(0)
        k_idx = k_dists.argmin(dim=1)
        v_idx = v_dists.argmin(dim=1)

        k_packed = key_packer.pack(k_idx.reshape(H_dim, C_dim))
        v_packed = val_packer.pack(v_idx.reshape(H_dim, C_dim))

        # Step 5: 构建结果
        return OutlierAwareCompressedKV(
            normal_key_bits=k_packed.cpu(),
            normal_val_bits=v_packed.cpu(),
            key_bits=self.key_bits,
            val_bits=self.val_bits,
            normal_shape=(B, H, S_normal, D),
            outlier_keys=k_outlier.cpu() if k_outlier is not None else None,
            outlier_values=v_outlier.cpu() if v_outlier is not None else None,
            outlier_indices=outlier_indices_t.cpu(),
            sigma_threshold=self.sigma_threshold,
            n_total=S,
            n_normal=S_normal,
            n_outlier=n_outlier,
        )

    @torch.no_grad()
    def decompress(
        self,
        state: OutlierAwareCompressedKV,
        centroid_k: torch.Tensor,
        centroid_v: torch.Tensor,
        key_packer,
        val_packer,
        unrot_fn,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        异常值感知解压。

        Returns:
            (keys, values): (B, H, S, D) 重建的 KV
        """
        B, H, S_normal, D = state.normal_shape
        n_total = state.n_total
        H_dim = B * H

        # 解压正常部分
        C_dim = S_normal
        k_idx = key_packer.unpack(state.normal_key_bits.to(key_packer.device), (H_dim, C_dim))
        v_idx = val_packer.unpack(state.normal_val_bits.to(val_packer.device), (H_dim, C_dim))

        k_normal = centroid_k[k_idx.reshape(-1)].reshape(B, H, S_normal, D)
        v_normal = centroid_v[v_idx.reshape(-1)].reshape(B, H, S_normal, D)

        # 逆旋转
        k_normal = unrot_fn(k_normal.float()).to(keys.dtype if 'keys' in dir() else k_normal.dtype)
        v_normal = unrot_fn(v_normal.float())

        # Step 2: 重建完整序列
        keys_out = torch.zeros(B, H, n_total, D, dtype=k_normal.dtype, device=k_normal.device)
        values_out = torch.zeros(B, H, n_total, D, dtype=v_normal.dtype, device=v_normal.device)

        # 填入正常 token
        normal_indices = state.outlier_indices
        # 计算正常位置（排除异常位置）
        all_idx = torch.arange(n_total)
        normal_mask = torch.ones(n_total, dtype=torch.bool)
        normal_mask[normal_indices.long()] = False
        normal_pos = all_idx[normal_mask]

        keys_out.index_copy_(dim=2, index=normal_pos, source=k_normal)
        values_out.index_copy_(dim=2, index=normal_pos, source=v_normal)

        # 填入异常值 token
        if state.n_outlier > 0 and state.outlier_keys is not None:
            outlier_k = state.outlier_keys.to(keys_out.device)
            outlier_v = state.outlier_values.to(values_out.device)
            outlier_idx = state.outlier_indices.to(keys_out.device)

            keys_out.index_copy_(dim=2, index=outlier_idx.long(), source=outlier_k)
            values_out.index_copy_(dim=2, index=outlier_idx.long(), source=outlier_v)

        return keys_out, values_out


# ===========================================================================
# 内存分析
# ===========================================================================

def outlier_protection_overhead(
    S: int,
    D: int,
    outlier_ratio: float = 0.003,
    key_bits: int = 4,
    val_bits: int = 2,
) -> Tuple[float, float]:
    """
    计算异常值保护机制的内存开销。

    Returns:
        (baseline_bytes, with_outlier_bytes, overhead_pct)
    """
    avg_bits = (key_bits + val_bits) / 2.0
    baseline = S * D * (avg_bits / 16.0) * 2  # 纯量化

    n_outlier = int(S * outlier_ratio)
    normal_S = S - n_outlier
    normal_bytes = normal_S * D * (avg_bits / 16.0) * 2
    outlier_bytes = n_outlier * D * 2 * 2  # K+V 原始 FP16
    index_bytes = S * 2  # outlier 位置标记

    with_outlier = normal_bytes + outlier_bytes + index_bytes

    overhead = (with_outlier - baseline) / baseline * 100
    return baseline, with_outlier, overhead
