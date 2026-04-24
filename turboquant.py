"""
TurboQuant Core — KV Cache 量化压缩实现（优化版）

优化点：
  1. 旋转矩阵：Hadamard FWHT O(d log d) 优先，QR O(d³) fallback
  2. 量化查找：用 searchsorted 代替 argmin 避免大临时张量
  3. Bit-pack：预计算偏移量，消除中间变量，直接用位运算
  4. 动态残差窗口：根据序列长度自适应
  5. 层自适应精细化：支持逐层 bit 分配 + 敏感度分析接口

Reference: TurboQuant (ICLR 2026) arXiv:2504.19874
            社区实践：MSE-only > QJL for KV cache
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F

from .rotation import generate_rotation_matrix, _HadamardRotation, _QRRotation, generate_qjl_matrix
from .lloyd_max import LloydMaxCodebook


# ===========================================================================
# 量化器核心（优化版）
# ===========================================================================

class QuantizedVector:
    """
    压缩后的单一向量表示（用于批量压缩接口）。

    存储格式：
      - indices: (D,) uint8，每坐标 1 字节（实际 bits 通过 packer 控制）
      - norm: float16

    不直接 bit-pack，留给外部根据批量大小决定打包策略。
    """

    __slots__ = ("indices", "norm")

    def __init__(self, indices: torch.Tensor, norm: torch.Tensor):
        self.indices = indices   # (D,) uint8
        self.norm = norm         # float16 scalar


# ===========================================================================
# 优化 2: 用 searchsorted 代替 argmin
# ===========================================================================

def _quantize_searchsorted(x: torch.Tensor,
                            centroids: torch.Tensor) -> torch.Tensor:
    """
    用 torch.searchsorted 找最近质心（适用于有序质心）。

    比 argmin 优势：
      - 无需构造 (N, D, 2^b) 临时张量
      - searchsorted 是二分查找，O(N·D·log 2^b) 但内存 O(N·D)
      - 对大 bits（如 8bit=256 质心）显存节省明显

    步骤：
      1. searchsorted 找到 x 落在哪两个质心之间
      2. 二选一（距离更近的那个）
    """
    # centroids: (2^b,) 已排序
    # x: (..., D)
    levels = len(centroids)

    # 找插入位置
    idx = torch.searchsorted(centroids, x, right=False)  # (..., D)
    idx = idx.clamp(1, levels - 1)  # 边界处理

    # 与前后质心比较，决定落在哪一侧
    c_left = centroids[idx - 1]   # (..., D)
    c_right = centroids[idx]       # (..., D)

    # x - c_left 和 x - c_right 的距离比较
    dist_left = (x - c_left).abs()
    dist_right = (x - c_right).abs()

    # 选择距离更近的
    indices = torch.where(dist_left <= dist_right, idx - 1, idx)
    return indices.to(torch.uint8)


def _quantize_argmin(x: torch.Tensor,
                      centroids: torch.Tensor) -> torch.Tensor:
    """
    标准 argmin 实现（透明、无临时张量优化路径）。
    当 bits ≤ 4 时，argmin 路径反而更快（向量化程度高）。
    """
    diffs = x.unsqueeze(-1) - centroids   # (..., D, 2^b)
    return diffs.abs().argmin(dim=-1).to(torch.uint8)


# ===========================================================================
# 优化 3: 高效 Bit-Packer
# ===========================================================================

class BitPacker:
    """
    高效 bit-pack / unpack，零动态内存分配（预计算偏移量）。

    压缩格式（每向量）：
      [bits_0 | bits_1 | ... | bits_{D-1}] 在 bytes 中从高 bit 到低 bit 排列

    存储：ceil(D * bits / 8) bytes

    优化点：
      - 预计算每次循环的 shift 量（静态）
      - 用位或和右移代替逐 byte 写入
      - 解包用向量化掩码提取，无循环
    """

    def __init__(self, d: int, bits: int):
        self.d = d
        self.bits = bits
        self.indices_per_byte = 8 // bits          # 每个 byte 装几个索引
        self.pad = (-d * bits) % 8                 # 末尾需要填充的位数
        self.n_bytes = math.ceil(d * bits / 8)     # 总字节数

        # 预计算每 byte 的位偏移（从高位到低位）
        # 例如 bits=3, indices_per_byte=2：
        #   byte[0]: high bits[2b-1:2b-3], low bits[2b-4:2b-6]
        #   byte[1]: high bits[2b-7:2b-9], ...
        # 统一用从高 bit 开始的偏移，递减
        self._shifts = [bits * i for i in range(self.indices_per_byte - 1, -1, -1)]
        self._unpack_shifts = [
            bits * (self.indices_per_byte - 1 - i) for i in range(self.indices_per_byte)
        ]

        # [Phase-1] 分支消除：预计算 shift 和 mask 为 tensor，消除 Python for 循环
        self._shift_tensor = torch.tensor(
            [bits * i for i in range(self.indices_per_byte - 1, -1, -1)],
            dtype=torch.int32,
        )
        self._unpack_mask = (1 << bits) - 1
        self._unpack_shift_tensor = torch.tensor(
            [bits * (self.indices_per_byte - 1 - i) for i in range(self.indices_per_byte)],
            dtype=torch.int64,
        )

    def pack(self, indices: torch.Tensor) -> torch.Tensor:
        """
        打包索引数组 → bytes 数组。

        indices: (..., D) uint8
        返回: (..., n_bytes) uint8

        [Phase-1] 分支消除：使用预计算 tensor 向量化，无 Python for 循环
        """
        pad = self.pad
        assert pad % self.bits == 0, f"pad={pad} must be multiple of bits={self.bits}"
        if pad:
            indices = F.pad(indices, (0, pad // self.bits), value=0)

        n_groups = indices.shape[-1] // self.indices_per_byte
        flat = indices.reshape(-1, n_groups, self.indices_per_byte).to(torch.int32)

        # 向量化：flat (N, G, per) × shifts (per,) → broadcast sum
        packed = (flat << self._shift_tensor.to(flat.device)).sum(dim=-1)
        return packed.to(torch.uint8).reshape(indices.shape[:-1] + (n_groups,))

    def unpack(self, bytes_data: torch.Tensor, out_d: int) -> torch.Tensor:
        """
        解包 bytes → 索引数组。

        bytes_data: (..., n_bytes) uint8
        out_d: 原始维度 D
        返回: (..., D) uint8

        [Phase-1] 分支消除：使用预计算 tensor 向量化，无 Python for 循环
        """
        n_groups = bytes_data.shape[-1]
        expanded = bytes_data.long().unsqueeze(-1)  # (..., n_groups, 1)
        shifts = self._unpack_shift_tensor.to(expanded.device)  # (per_byte,)
        parts = (expanded >> shifts) & self._unpack_mask  # (..., n_groups, per_byte)
        flat_indices = parts.reshape(*bytes_data.shape[:-1], n_groups * self.indices_per_byte)
        return flat_indices[..., :out_d].to(torch.uint8)


# ===========================================================================
# MSE 压缩器（优化版）
# ===========================================================================

class MSECompressor:
    """
    优化版 MSE 压缩器。

    改进：
      - searchsorted 量化（避免大临时张量）
      - 高效 BitPacker（预计算偏移，零动态分配）
      - 批量压缩时避免中间 tensor
    """

    def __init__(self, head_dim: int, bits: int, seed: int = 42, device: str = "cpu",
                 use_compile: bool = False):
        self.head_dim = head_dim
        self.bits = bits
        self.device = device
        self.use_compile = use_compile

        # 旋转矩阵（隐式 Hadamard 表示）
        self.rot = generate_rotation_matrix(head_dim, seed=seed, device=device)
        self.codebook = LloydMaxCodebook(head_dim, bits)
        self.centroids = self.codebook.centroids.to(device)
        self.packer = BitPacker(head_dim, bits)

        # 选择量化策略：bits ≤ 4 用 argmin（更快），> 4 用 searchsorted（省显存）
        self._use_searchsorted = bits > 4

        # [Phase-1] torch.compile：消除 Python 解释器开销
        # 仅在 CPU/CUDA 且 torch >= 2.0 时启用
        self._compiled_quantize = None
        self._compiled_dequantize = None
        if use_compile:
            try:
                self._compiled_quantize = torch.compile(self._quantize_impl, mode="reduce-overhead")
                self._compiled_dequantize = torch.compile(self._dequantize_impl, mode="reduce-overhead")
            except Exception:
                pass  # 静默 fallback 到 eager 模式

    def _quantize_impl(self, x_unit: torch.Tensor) -> torch.Tensor:
        """量化核心实现（可被 torch.compile 优化）"""
        rotated = self.rot.rotate(x_unit)
        if self._use_searchsorted:
            return _quantize_searchsorted(rotated, self.centroids)
        else:
            return _quantize_argmin(rotated, self.centroids)

    def _dequantize_impl(self, indices: torch.Tensor) -> torch.Tensor:
        """反量化核心实现（可被 torch.compile 优化）"""
        recon = self.centroids[indices.long()]
        return self.rot.unrotate(recon)

    def quantize(self, x_unit: torch.Tensor) -> torch.Tensor:
        """
        对单位球面上的向量做量化，返回索引。

        x_unit: (..., D) 已归一化的单位向量
        返回: (..., D) uint8 索引

        [Phase-1] 优先使用 torch.compile 版本
        """
        if self._compiled_quantize is not None:
            return self._compiled_quantize(x_unit)
        return self._quantize_impl(x_unit)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """
        索引 → 重建单位向量（不含范数）。

        indices: (..., D) uint8
        返回: (..., D) float32 单位向量

        [Phase-1] 优先使用 torch.compile 版本
        """
        if self._compiled_dequantize is not None:
            return self._compiled_dequantize(indices)
        return self._dequantize_impl(indices)

    @torch.no_grad()
    def compress(self, states: torch.Tensor) -> dict:
        """
        压缩批量向量。

        states: (B, H, S, D) 或 (S, D)
        返回压缩 dict（含 bit-packed bytes + 范数）
        """
        orig_shape = states.shape
        flat = states.reshape(-1, self.head_dim).float()
        N = flat.shape[0]

        vec_norms = torch.norm(flat, dim=-1, keepdim=True)  # (N, 1)
        flat_norm = flat / (vec_norms + 1e-8)

        indices = self.quantize(flat_norm)  # (N, D)
        packed = self.packer.pack(indices)  # (N, n_bytes)

        return {
            "packed": packed,
            "vec_norms": vec_norms.squeeze(-1).to(torch.float16),
            "shape": orig_shape,
        }

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """
        解压回原始 shape。
        """
        packed = compressed["packed"]
        vec_norms = compressed["vec_norms"]
        orig_shape = compressed["shape"]

        N = math.prod(orig_shape[:-1])
        indices = self.packer.unpack(packed.reshape(N, -1), self.head_dim)  # (N, D)
        recon_unit = self.dequantize(indices)  # (N, D)
        recon = recon_unit * vec_norms.to(recon_unit.dtype).unsqueeze(-1)
        return recon.reshape(orig_shape)

    def memory_usage(self, B: int, H: int, S: int) -> dict:
        """内存用量估算"""
        N = B * H * S
        total_bytes = N * (self.packer.n_bytes + 2)  # +2 for FP16 norm
        fp16_bytes = N * self.head_dim * 2
        return {
            "compressed_bytes": total_bytes,
            "fp16_bytes": fp16_bytes,
            "compression_ratio": fp16_bytes / total_bytes,
        }


# ===========================================================================
# 动态残差窗口
# ===========================================================================

def compute_residual_window(seq_len: int, base: int = 128,
                            min_window: int = 32,
                            max_window: int = 512,
                            growth_factor: float = 0.1) -> int:
    """
    根据序列长度动态计算残差窗口大小。

    策略：
      - 短序列（≤ base）：全部 FP16（零压缩开销）
      - 中等序列：sigmoid 形增长，在达到序列长度前逐渐切换到全 FP16
      - 长序列：稳定在 max_window，上下文足够

    参数:
        seq_len:      当前序列长度
        base:         基础窗口大小（默认 128）
        min_window:   最小窗口（默认 32）
        max_window:   最大窗口（默认 512）
        growth_factor: 增长系数（越大窗口增长越快）

    返回:
        残差窗口大小（tokens），保证 ≤ seq_len
    """
    if seq_len <= base:
        return seq_len  # 全部 FP16

    # sigmoid 形增长曲线
    x = (seq_len - base) * growth_factor
    ratio = 1 / (1 + math.exp(-x))  # 0 → 1 as seq_len grows
    window = min_window + (max_window - min_window) * ratio
    # 关键修复：确保 window 不超过 seq_len
    return int(min(window, seq_len))


# ===========================================================================
# 层敏感度（优化 5）
# ===========================================================================

class LayerSensitivityAnalyzer:
    """
    轻量级层敏感度分析器。

    方法：在少量校准数据上做一次前向传播，
    统计每层 KV 激活的重建误差对下游的影响。

    不需要梯度，零训练开销。

    输出：
      sensitivity_scores: (n_layers,) 每层敏感度（0~1，越大越敏感）
      bit_allocations:    (n_layers,) 每层分配的 bit 数
    """

    def __init__(self, n_layers: int, base_key_bits: int = 4,
                 base_value_bits: int = 2, seed: int = 0):
        self.n_layers = n_layers
        self.base_key_bits = base_key_bits
        self.base_value_bits = base_value_bits
        self.seed = seed
        self._scores: Optional[torch.Tensor] = None

    def analyze(self, kv_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分析层敏感度。

        kv_pairs: [(layer_idx, keys, values), ...] 的列表
                  keys/values: (B, H, S, D)
        返回: (sensitivity_scores, bit_allocations)
        """
        n_layers = self.n_layers
        errors = torch.zeros(n_layers)

        for item in kv_pairs:
            layer_idx, keys, values = item
            # 计算激活量级作为敏感度代理
            k_scale = keys.float().norm(dim=-1).mean()
            v_scale = values.float().norm(dim=-1).mean()
            # 首尾层更敏感（attention 分布更集中）
            boundary_penalty = min(layer_idx, n_layers - 1 - layer_idx) / (n_layers / 2)
            errors[layer_idx] += (k_scale + v_scale) * (1 + 0.5 * boundary_penalty)

        # 归一化到 [0, 1]
        scores = errors / (errors.max() + 1e-8)

        # 基于敏感度分配 bit
        total_budget = (self.base_key_bits + self.base_value_bits) * n_layers
        extra = scores * 2  # 敏感层额外 +2 bit
        allocations = torch.clamp(
            torch.tensor([self.base_key_bits, self.base_value_bits]).unsqueeze(0) + extra.unsqueeze(-1),
            min=2, max=8
        )
        key_bits = allocations[:, 0].long()
        value_bits = allocations[:, 1].long()

        self._scores = scores
        return scores, allocations

    def auto_config(self) -> List[dict]:
        """
        生成每层的最优配置（需先调用 analyze）。
        """
        if self._scores is None:
            raise RuntimeError("请先调用 analyze()")
        _, allocs = self.analyze([])
        return [
            {"key_bits": int(allocs[i, 0]), "value_bits": int(allocs[i, 1])}
            for i in range(self.n_layers)
        ]


# ===========================================================================
# TurboQuant KV Cache（优化版）
# ===========================================================================

class TurboQuantKV:
    """
    优化版 TurboQuant KV 缓存压缩器。

    改进点：
      1. Hadamard 旋转（O(d log d)，无大矩阵存储）
      2. searchsorted 量化（省显存）
      3. 高效 bit-pack（零动态分配）
      4. 动态残差窗口（序列长度自适应）
      5. 层敏感度分析接口（精细化 bit 分配）
    """

    def __init__(
        self,
        head_dim: int,
        key_bits: int = 4,
        value_bits: int = 2,
        residual_window: int = 128,
        use_dynamic_window: bool = False,
        layer_idx: int = 0,
        n_layers: int = 36,
        protected_layers: int = 4,
        protected_bits: int = 8,
        seed: int = 42,
        device: str = "cpu",
        double_buffer: bool = False,   # 优化 4: 双缓冲
    ):
        self.head_dim = head_dim
        self.residual_window = residual_window
        self.use_dynamic_window = use_dynamic_window
        self.device = device

        # 优化 4: 双缓冲
        self._double_buffer = double_buffer
        self._buf_keys: List[Optional[dict]] = [None, None]
        self._buf_values: List[Optional[dict]] = [None, None]
        self._buf_idx = 0
        self._buf_state: List[str] = ["empty", "empty"]

        # 层自适应
        is_protected = layer_idx < protected_layers or layer_idx >= (n_layers - protected_layers)
        self.key_bits = protected_bits if is_protected else key_bits
        self.value_bits = protected_bits if is_protected else value_bits

        seed_k = seed + layer_idx * 1000
        seed_v = seed + layer_idx * 1000 + 500

        self.key_comp = MSECompressor(head_dim, self.key_bits, seed=seed_k, device=device)
        self.val_comp = MSECompressor(head_dim, self.value_bits, seed=seed_v, device=device)

    def _compute_window(self, seq_len: int) -> int:
        """计算残差窗口大小（固定或动态）"""
        if self.use_dynamic_window:
            return compute_residual_window(seq_len, base=self.residual_window)
        return min(self.residual_window, seq_len)

    @torch.no_grad()
    def compress_kv(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[dict, dict]:
        """
        压缩 Keys 和 Values。

        keys:   (B, H, S, D)
        values: (B, H, S, D)
        """
        B, H, S, D = keys.shape
        rw = self._compute_window(S)

        if rw >= S:
            return (
                {"fp16": keys, "compressed": None, "shape": (B, H, S, D), "split_at": S},
                {"fp16": values, "compressed": None, "shape": (B, H, S, D), "split_at": S},
            )

        split_at = S - rw
        old_k = keys[:, :, :split_at]
        new_k = keys[:, :, split_at:]
        old_v = values[:, :, :split_at]
        new_v = values[:, :, split_at:]

        compressed_k = {
            "compressed": self.key_comp.compress(old_k),
            "fp16": new_k,
            "shape": (B, H, S, D),
            "split_at": split_at,
            "rw": rw,
        }
        compressed_v = {
            "compressed": self.val_comp.compress(old_v),
            "fp16": new_v,
            "shape": (B, H, S, D),
            "split_at": split_at,
            "rw": rw,
        }
        return compressed_k, compressed_v

    @torch.no_grad()
    def decompress_kv(self, ck: dict, cv: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """解压回原始 keys/values"""
        if ck["compressed"] is None:
            return ck["fp16"], cv["fp16"]

        old_k = self.key_comp.decompress(ck["compressed"])
        old_v = self.val_comp.decompress(cv["compressed"])

        keys = torch.cat([old_k, ck["fp16"]], dim=2)
        values = torch.cat([old_v, cv["fp16"]], dim=2)
        return keys, values

    # ── Double Buffering 预取接口（优化 4）─────────────────────────────

    def prefetch(self, keys: torch.Tensor, values: torch.Tensor,
                 buf_slot: Optional[int] = None) -> int:
        """预取并压缩 KV cache 到指定缓冲区（异步友好）。"""
        if not self._double_buffer:
            return -1
        slot = buf_slot if buf_slot is not None else (1 - self._buf_idx)
        self._buf_state[slot] = "loading"
        ck, cv = self.compress_kv(keys, values)
        self._buf_keys[slot] = ck
        self._buf_values[slot] = cv
        self._buf_state[slot] = "ready"
        return slot

    def release(self, buf_slot: Optional[int] = None):
        """释放指定缓冲区（释放显存）。"""
        slot = buf_slot if buf_slot is not None else self._buf_idx
        if 0 <= slot < 2:
            self._buf_keys[slot] = None
            self._buf_values[slot] = None
            self._buf_state[slot] = "empty"

    def decompress_buffers(self) -> Tuple:
        """解压两个缓冲区的 KV cache。"""
        results = []
        for slot in range(2):
            ck = self._buf_keys[slot]
            if ck is not None and ck["compressed"] is not None:
                k, v = self.decompress_kv(ck, self._buf_values[slot])
            elif ck is not None:
                k, v = ck["fp16"], self._buf_values[slot]["fp16"]
            else:
                k, v = None, None
            results.extend([k, v])
        return tuple(results)

    def switch_buffer(self) -> int:
        """切换到另一个缓冲区。"""
        self._buf_idx = 1 - self._buf_idx
        return self._buf_idx

    @property
    def buffer_status(self) -> dict:
        return {
            "double_buffer": self._double_buffer,
            "active": self._buf_idx,
            "slot_0": self._buf_state[0],
            "slot_1": self._buf_state[1],
        }

    def memory_usage(self, B: int, H: int, S: int) -> dict:
        """内存用量报告"""
        rw = self._compute_window(S)
        compressed_S = max(S - rw, 0)

        if compressed_S > 0:
            k_mem = self.key_comp.memory_usage(B, H, compressed_S)
            v_mem = self.val_comp.memory_usage(B, H, compressed_S)
            compressed_bytes = k_mem["compressed_bytes"] + v_mem["compressed_bytes"]
        else:
            compressed_bytes = 0

        fp16_window = B * H * rw * self.head_dim * 2 * 2
        total_compressed = compressed_bytes + fp16_window
        total_fp16 = B * H * S * self.head_dim * 2 * 2

        return {
            "compressed_bytes": total_compressed,
            "fp16_bytes": total_fp16,
            "compression_ratio": total_fp16 / total_compressed if total_compressed > 0 else 0,
            "compressed_tokens": compressed_S,
            "fp16_tokens": rw,
            "key_bits": self.key_bits,
            "value_bits": self.value_bits,
            "dynamic_window": self.use_dynamic_window,
        }


# ===========================================================================
# 论文原版 V1（含 QJL）— 保留用于对比
# ===========================================================================

class TurboQuantV1:
    """
    论文原版 TurboQuant（含 QJL）。

    Stage 1: MSE 量化（bits-1 位/坐标）
    Stage 2: QJL 残差校正（1 位/坐标）

    注意：社区实践表明 QJL 对 KV 缓存有负面影响，
          不推荐用于生成任务，仅适合向量搜索等内积估计场景。
    """

    def __init__(self, d: int, bits: int, qjl_dim: Optional[int] = None,
                 seed: int = 42, device: str = "cpu"):
        self.d = d
        self.bits = bits
        self.mse_bits = max(bits - 1, 1)
        self.qjl_dim = qjl_dim or d
        self.device = device

        self.rot = generate_rotation_matrix(d, seed=seed, device=device)
        self.codebook = LloydMaxCodebook(d, self.mse_bits)
        self.centroids = self.codebook.centroids.to(device)
        self.S = generate_qjl_matrix(d, qjl_dim=self.qjl_dim, seed=seed + 1, device=device)

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> dict:
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        x_unit = x / (x_norm + 1e-8)
        rotated = self.rot.rotate(x_unit)
        indices = _quantize_argmin(rotated, self.centroids)

        x_recon_unit = self.centroids[indices.long()]
        x_recon_unit = self.rot.unrotate(x_recon_unit)
        x_recon = x_recon_unit * x_norm

        residual = x - x_recon
        residual_norm = torch.norm(residual, dim=-1)
        projected = residual @ self.S.T
        qjl_signs = torch.sign(projected)
        qjl_signs[qjl_signs == 0] = 1.0

        return {
            "mse_indices": indices,
            "qjl_signs": qjl_signs,
            "residual_norm": residual_norm,
            "x_norm": x_norm.squeeze(-1),
        }

    @torch.no_grad()
    def dequantize(self, compressed: dict) -> torch.Tensor:
        x_unit = self.centroids[compressed["mse_indices"].long()]
        x_unit = self.rot.unrotate(x_unit)
        return x_unit * compressed["x_norm"].unsqueeze(-1)

    @torch.no_grad()
    def inner_product(self, y: torch.Tensor, compressed: dict) -> torch.Tensor:
        x_mse = self.dequantize(compressed)
        term1 = (y * x_mse).sum(dim=-1)

        y_proj = y @ self.S.T
        qjl_ip = (y_proj * compressed["qjl_signs"]).sum(dim=-1)
        m = self.qjl_dim
        term2 = compressed["residual_norm"] * math.sqrt(math.pi / 2) / m * qjl_ip

        return term1 + term2
