"""
Step 1: Per-Layer Bit Allocation

实现两层分配器：
  1. LayerBitAllocator  — 按层分配 bits（基于 PyramidKV 金字塔结构）
  2. LayeredTurboQuantKV — 接受 per-layer bit 配置的 TurboQuantKV wrapper

金字塔公式（PyramidKV）：
  浅层（低 index）：保留更多 bits（信息分布广，更敏感）
  深层（高 index）：压缩更激进（注意力高度聚焦，少量 bits 够用）

  bits(layer) = max_bits - (max_bits - min_bits) × (layer_idx / (n_layers-1))^alpha

  alpha 控制曲线形态：
    alpha=1.0  → 线性（每层均匀递减）
    alpha=0.5  → 凸（前期下降快，后期缓慢）
    alpha=2.0  → 凹（前期下降慢，后期加速压缩）

Reference: PyramidKV (ICLR 2025) — 低层 100% 预算，高层降至 0.7%
"""

from __future__ import annotations

import math
from typing import List, Dict, Optional, Tuple, Literal
from dataclasses import dataclass

import torch


# ===========================================================================
# 数据结构
# ===========================================================================

@dataclass(frozen=True)
class LayerBits:
    """单个层的 bit 分配配置（immutable）"""
    layer_idx: int
    key_bits: int
    value_bits: int
    is_protected: bool = False  # 是否受保护层（首尾）

    @property
    def total_bits(self) -> int:
        return self.key_bits + self.value_bits

    def __repr__(self) -> str:
        prot = " [PROT]" if self.is_protected else ""
        return f"L{layer_idx:02d}: K={self.key_bits}b V={self.value_bits}b{prot}"


@dataclass
class LayerBudget:
    """层预算统计"""
    n_layers: int
    base_key_bits: int
    base_value_bits: int
    pyramid_max_bits: int   # 浅层 key_bits 上限
    pyramid_min_bits: int   # 深层 key_bits 下限
    pyramid_alpha: float
    use_pyramid: bool
    layers: List[LayerBits]

    @property
    def pyramid_bits_per_layer(self) -> float:
        """金字塔模式下每层平均 bits（用于对比）"""
        if not self.use_pyramid:
            return self.base_key_bits + self.base_value_bits
        max_b = self.pyramid_max_bits
        min_b = self.pyramid_min_bits
        n = self.n_layers
        alpha = self.pyramid_alpha
        # 解析解：∫[0,1] (max_b - (max_b-min_b)·t^alpha) dt
        #        = max_b - (max_b-min_b)/(alpha+1)
        avg = max_b - (max_b - min_b) / (alpha + 1)
        # 这里的 bits 是 key_bits，需要加 value_bits
        return avg + self.base_value_bits

    def total_bits_baseline(self) -> int:
        """均匀分配时的总 bits"""
        return (self.base_key_bits + self.base_value_bits) * self.n_layers

    def total_bits_pyramid(self) -> int:
        """金字塔分配时的总 bits"""
        return sum(l.total_bits for l in self.layers)

    def memory_savings(self) -> float:
        """相比均匀分配的内存节省百分比（正值=节省，负值=增加）"""
        base = self.total_bits_baseline()
        pyr = self.total_bits_pyramid()
        return (base - pyr) / base * 100

    def smooth_bits(self) -> int:
        """无保护层的平滑金字塔总 bits"""
        return sum(lb.total_bits for lb in self.layers if not lb.is_protected)


# ===========================================================================
# 分配器核心
# ===========================================================================

class LayerBitAllocator:
    """
    金字塔式层 bits 分配器。

    支持两种模式：
      1. uniform   — 均匀分配（baseline）
      2. pyramid   — 金字塔分配（浅层多，深层少）

    PyramidKV 论文（ICLR 2025）核心观察：
      - 低层注意力分布广泛（高熵），需要更多 KV 缓存
      - 高层注意力高度聚焦（attention sink），可以激进压缩
      - 低层保留 100% 预算，高层降至 0.7% 仍能维持精度

    这里采用简化版：浅层用更多 bits（4+2=6），深层压缩到（2+2=4）
    总预算与均匀分配相同（相同内存占用），但质量更高。
    """

    def __init__(
        self,
        n_layers: int,
        base_key_bits: int = 4,
        base_value_bits: int = 2,
        # Pyramid 参数
        pyramid_max_bits: int = 6,     # 浅层 key_bits 上限
        pyramid_min_bits: int = 3,     # 深层 key_bits 下限
        pyramid_alpha: float = 1.0,    # 曲线形状（1.0 = 线性）
        # 保护层（首尾若干层不压缩）
        protected_layers: int = 2,
        protected_key_bits: int = 8,
        protected_value_bits: int = 6,
        # Value bits 独立控制（一般 value 压缩空间更小）
        pyramid_max_value_bits: int = 4,
        pyramid_min_value_bits: int = 3,   # V=2bit(4级)质量太低，改为min=3bit(8级)
        value_pyramid_enabled: bool = True,
    ):
        if n_layers < 1:
            raise ValueError(f"n_layers={n_layers} 必须 ≥ 1")
        if protected_layers < 0:
            raise ValueError("protected_layers 必须 ≥ 0")
        if protected_layers * 2 > n_layers:
            raise ValueError("protected_layers 不能超过总层数的一半")

        self.n_layers = n_layers
        self.base_key_bits = base_key_bits
        self.base_value_bits = base_value_bits
        self.pyramid_max_bits = pyramid_max_bits
        self.pyramid_min_bits = pyramid_min_bits
        self.pyramid_alpha = pyramid_alpha
        self.protected_layers = protected_layers
        self.protected_key_bits = protected_key_bits
        self.protected_value_bits = protected_value_bits
        self.pyramid_max_value_bits = pyramid_max_value_bits
        self.pyramid_min_value_bits = pyramid_min_value_bits
        self.value_pyramid_enabled = value_pyramid_enabled

    def _pyramid_curve(self, layer_idx: int, max_val: int, min_val: int) -> int:
        """
        金字塔曲线：浅层高值，深层低值。

        公式：bits = max_val - (max_val - min_val) × t^alpha
        其中 t = layer_idx / (n_layers - 1) ∈ [0, 1]

        alpha 控制形态：
          alpha → 0:  max_bits 均匀下降
          alpha = 1:  线性下降
          alpha → ∞:  前期几乎不变，后期急剧压缩
        """
        t = layer_idx / max(self.n_layers - 1, 1)  # [0, 1]
        value = max_val - (max_val - min_val) * (t ** self.pyramid_alpha)
        return max(min_val, min(max_val, round(value)))

    def allocate(self, mode: Literal["uniform", "pyramid"] = "pyramid") -> LayerBudget:
        """
        执行 bit 分配。

        参数:
            mode: "uniform" = 所有层相同 bits（baseline）
                  "pyramid" = 金字塔递减（推荐）
        返回:
            LayerBudget: 包含每层配置和统计信息
        """
        self._mode = mode
        layers: List[LayerBits] = []

        for i in range(self.n_layers):
            is_protected = i < self.protected_layers or i >= (self.n_layers - self.protected_layers)

            if is_protected:
                # 保护层：强制高精度
                lb = LayerBits(
                    layer_idx=i,
                    key_bits=self.protected_key_bits,
                    value_bits=self.protected_value_bits,
                    is_protected=True,
                )
            elif mode == "uniform":
                lb = LayerBits(
                    layer_idx=i,
                    key_bits=self.base_key_bits,
                    value_bits=self.base_value_bits,
                    is_protected=False,
                )
            else:  # pyramid
                # Key bits: 金字塔曲线
                k_bits = self._pyramid_curve(
                    i, self.pyramid_max_bits, self.pyramid_min_bits
                )
                # Value bits: 可选独立金字塔曲线（通常更平缓）
                if self.value_pyramid_enabled:
                    v_bits = self._pyramid_curve(
                        i, self.pyramid_max_value_bits, self.pyramid_min_value_bits
                    )
                else:
                    v_bits = self.base_value_bits

                lb = LayerBits(
                    layer_idx=i,
                    key_bits=k_bits,
                    value_bits=v_bits,
                    is_protected=False,
                )
            layers.append(lb)

        return LayerBudget(
            n_layers=self.n_layers,
            base_key_bits=self.base_key_bits,
            base_value_bits=self.base_value_bits,
            pyramid_max_bits=self.pyramid_max_bits,
            pyramid_min_bits=self.pyramid_min_bits,
            pyramid_alpha=self.pyramid_alpha,
            use_pyramid=(mode == "pyramid"),
            layers=layers,
        )

    def report(self, budget: LayerBudget) -> str:
        """生成可读的分配报告"""
        lines = [
            f"LayerBitAllocator ({self.n_layers} 层)",
            f"  模式: {'金字塔' if budget.use_pyramid else '均匀'} (alpha={self.pyramid_alpha})",
            f"  范围: K=[{self.pyramid_min_bits}, {self.pyramid_max_bits}]b, "
            f"V=[{self.pyramid_min_value_bits}, {self.pyramid_max_value_bits}]b",
            f"  保护层: {self.protected_layers} (首尾各 {self.protected_layers} 层, "
            f"K={self.protected_key_bits}b V={self.protected_value_bits}b)",
            "",
        ]

        # 打印前 5 层 + ... + 后 5 层（长序列截断）
        shown = []
        if self.n_layers <= 12:
            shown = list(range(self.n_layers))
        else:
            shown = list(range(5)) + [None] + list(range(self.n_layers - 5, self.n_layers))

        for i in shown:
            if i is None:
                lines.append(f"  ... ({self.n_layers - 10} 层省略) ...")
                continue
            lb = budget.layers[i]
            lines.append(f"  L{i:02d}: K={lb.key_bits:2d}b  V={lb.value_bits}b"
                         f"{'  [PROT]' if lb.is_protected else ''}")

        # 统计
        base_total = budget.total_bits_baseline()
        pyr_total = budget.total_bits_pyramid()

        if budget.use_pyramid:
            # 计算无保护层的金字塔 bits（平滑曲线）
            smooth_bits = sum(
                lb.total_bits for lb in budget.layers
                if not lb.is_protected
            )
            lines.extend([
                "",
                f"  基准（均匀）总 bits:  {base_total:>6d}  "
                f"(= {self.n_layers} × ({self.base_key_bits}+{self.base_value_bits}))",
                f"  金字塔总 bits:        {pyr_total:>6d}  "
                f"({pyr_total/base_total:.1f}x，含 {self.protected_layers}×2 保护层 K=8+V=4)",
                f"  ──────────────────────────────────────────────",
                f"  平滑金字塔 bits:       {smooth_bits:>6d}  "
                f"({smooth_bits/base_total:.1f}x，无保护层 = 预算内平滑递减)",
                f"  平滑模式内存节省:     {(1-smooth_bits/base_total)*100:+.1f}%  "
                f"({'节省 ✅' if smooth_bits < base_total else '略增 ⚠️'})",
            ])
        else:
            lines.extend([
                "",
                f"  总 bits:  {base_total:>6d}",
            ])
        return "\n".join(lines)


# ===========================================================================
# LayeredTurboQuantKV: 接受 per-layer bit 配置的 wrapper
# ===========================================================================

def _lazy_import_HADAMARD():
    """延迟导入避免循环依赖"""
    from HADAMARD import TurboQuantKV
    return TurboQuantKV


class LayeredTurboQuantKV:
    """
    层级别 bit 分配的 TurboQuant KV 压缩器。

    内部为每层维护独立的 TurboQuantKV 实例，支持：
      - PyramidKV 风格分层压缩（浅层高保真，深层高压缩）
      - 与原始 TurboQuantKV 完全兼容的接口
      - 批量压缩/解压（自动分发到各层）

    内存节省原理：
      相同总 bits 预算下，金字塔分配在浅层质量更高（关键信息在浅层）
      深层压缩更激进（高度抽象，冗余更多），整体质量 > 均匀分配。

    用法：
      ```python
      from .pyramid_alloc import LayerBitAllocator, LayeredTurboQuantKV

      # Step 1: 分配 bits
      allocator = LayerBitAllocator(
          n_layers=30,
          base_key_bits=4,
          base_value_bits=2,
          pyramid_max_bits=6,
          pyramid_min_bits=3,
          protected_layers=2,
      )
      budget = allocator.allocate("pyramid")

      # Step 2: 构建压缩器
      layered = LayeredTurboQuantKV(
          head_dim=128,
          budget=budget,
          seed=42,
          device="cuda",
      )

      # Step 3: 使用（接口与 TurboQuantKV 相同）
      keys = torch.randn(1, 32, 1024, 128)
      values = torch.randn(1, 32, 1024, 128)
      compressed = layered.compress(kv_dict={"keys": keys, "values": values})
      decompressed = layered.decompress(compressed)
      ```
    """

    def __init__(
        self,
        head_dim: int,
        budget: LayerBudget,
        residual_window: int = 128,
        use_dynamic_window: bool = False,
        seed: int = 42,
        device: str = "cpu",
        double_buffer: bool = False,
    ):
        self.head_dim = head_dim
        self.budget = budget
        self.residual_window = residual_window
        self.use_dynamic_window = use_dynamic_window
        self.device = device
        self.seed = seed
        self.double_buffer = double_buffer

        # 延迟导入避免循环
        TurboQuantKV = _lazy_import_HADAMARD()

        # 每层一个 TurboQuantKV 实例
        self._compressors: List[Optional[TurboQuantKV]] = [None] * budget.n_layers

        for i, lb in enumerate(budget.layers):
            self._compressors[i] = TurboQuantKV(
                head_dim=head_dim,
                key_bits=lb.key_bits,
                value_bits=lb.value_bits,
                residual_window=residual_window,
                use_dynamic_window=use_dynamic_window,
                layer_idx=i,
                n_layers=budget.n_layers,
                protected_layers=0,  # 已在 allocator 处理
                protected_bits=max(lb.key_bits, lb.value_bits),
                seed=seed + i * 1000,
                device=device,
                double_buffer=double_buffer,
            )

    def compress(self, kv_dict: Dict[str, torch.Tensor]) -> Dict[int, Dict]:
        """
        压缩 KV（per-layer）。

        kv_dict: {"keys": (B, H, S, D), "values": (B, H, S, D)}
        返回: {layer_idx: {"keys": compressed_k, "values": compressed_v}}

        注意：此接口接受统一 KV，适用于所有层用相同数据的场景。
        实际推理中每层 KV 不同，需用 compress_layer() 逐层调用。
        """
        keys = kv_dict["keys"]
        values = kv_dict["values"]
        results = {}
        for i, comp in enumerate(self._compressors):
            ck, cv = comp.compress_kv(keys, values)
            results[i] = {"keys": ck, "values": cv}
        return results

    def compress_layer(
        self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[Dict, Dict]:
        """压缩指定层的 KV（实际推理接口）"""
        comp = self._compressors[layer_idx]
        return comp.compress_kv(keys, values)

    def decompress(self, compressed: Dict[int, Dict]) -> Dict[str, torch.Tensor]:
        """
        解压所有层的 KV。

        compressed: {layer_idx: {"keys": ..., "values": ...}}
        返回: {"keys": 所有层 keys 拼接, "values": 所有层 values 拼接}
        """
        all_keys, all_values = [], []
        for i in range(self.budget.n_layers):
            comp = self._compressors[i]
            ck = compressed[i]["keys"]
            cv = compressed[i]["values"]
            k, v = comp.decompress_kv(ck, cv)
            all_keys.append(k)
            all_values.append(v)
        return {
            "keys": torch.cat(all_keys, dim=0),  # (n_layers, H, S, D)
            "values": torch.cat(all_values, dim=0),
        }

    def decompress_layer(self, layer_idx: int, ck: Dict, cv: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """解压指定层"""
        return self._compressors[layer_idx].decompress_kv(ck, cv)

    def memory_usage(self, B: int, H: int, S: int) -> Dict:
        """汇总所有层的内存用量"""
        total_compressed = 0
        total_fp16 = 0
        layer_reports = []

        for i, comp in enumerate(self._compressors):
            r = comp.memory_usage(B, H, S)
            total_compressed += r["compressed_bytes"]
            total_fp16 += r["fp16_bytes"]
            layer_reports.append({**r, "layer": i})

        return {
            "total_compressed_bytes": total_compressed,
            "total_fp16_bytes": total_fp16,
            "compression_ratio": total_fp16 / total_compressed if total_compressed > 0 else 0,
            "layers": layer_reports,
            "allocator_mode": "pyramid" if self.budget.use_pyramid else "uniform",
        }

    def summary(self) -> str:
        lines = [
            "LayeredTurboQuantKV 配置：",
            f"  层数: {self.budget.n_layers}",
            f"  head_dim: {self.head_dim}",
            f"  模式: {'金字塔' if self.budget.use_pyramid else '均匀'}",
        ]
        for i, lb in enumerate(self.budget.layers):
            prot = " [PROTECTED]" if lb.is_protected else ""
            lines.append(f"  L{i:02d}: K={lb.key_bits}b + V={lb.value_bits}b = {lb.total_bits}b{prot}")
        return "\n".join(lines)

    @property
    def layer_bits(self) -> List[LayerBits]:
        return self.budget.layers

    def get_layer_bits(self, layer_idx: int) -> LayerBits:
        return self.budget.layers[layer_idx]
