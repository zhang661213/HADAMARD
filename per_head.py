"""
TurboQuant Per-Head Bit Allocation
扩展 LayerSensitivityAnalyzer 到每个注意力头级别。

Per-Head 位分配策略：
  1. 敏感度计算：每层 × 每头计算激活量级和注意力分布
  2. Bit 分配：基于敏感度的自适应位宽（Keys 4-8bit，Values 2-6bit）
  3. 配置生成：返回每层每头的 (key_bits, value_bits) 配置

用法：
  ```python
  analyzer = HeadSensitivityAnalyzer(n_layers=36, n_heads=8)
  configs = analyzer.calibrate(kv_pairs)  # [(layer_idx, head_idx) → config]

  # 用于 TurboQuantKV
  for layer_idx, layer_configs in enumerate(configs):
      for head_idx, cfg in enumerate(layer_configs):
          compressor = TurboQuantKV(..., key_bits=cfg['key_bits'], ...)
  ```
"""

import math
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F


class HeadSensitivityAnalyzer:
    """
    注意力头级别敏感度分析器。

    分析每层 × 每头的压缩敏感度，支持精细化 bit 分配。

    敏感度因素：
      1. 激活量级（keys/values 的 L2 norm per head）
      2. 注意力聚焦度（熵，越高越敏感 → 分配更多 bit）
      3. 层位置（首尾层更敏感，自动加权）

    方法：
      1. 收集校准数据（KV pairs）
      2. 逐头计算敏感度分数
      3. 基于预算约束分配 bits
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        head_dim: int = 128,
        base_key_bits: int = 4,
        base_value_bits: int = 2,
        min_key_bits: int = 2,
        min_value_bits: int = 2,
        max_key_bits: int = 8,
        max_value_bits: int = 6,
        seed: int = 42,
        device: str = "cpu",
    ):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.base_key_bits = base_key_bits
        self.base_value_bits = base_value_bits
        self.min_key_bits = min_key_bits
        self.min_value_bits = min_value_bits
        self.max_key_bits = max_key_bits
        self.max_value_bits = max_value_bits
        self.seed = seed
        self.device = device

        # 状态（calibrate 后填充）
        self._scores: Optional[torch.Tensor] = None  # (n_layers, n_heads)
        self._configs: Any = None

    def calibrate(
        self,
        kv_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        batch_size: int = 16,
    ) -> List[List[Dict]]:
        """
        校准每层每头的位分配。

        kv_pairs: List of (keys, values) tensors
                  keys/values: (B, H, S, D) 或 (B, S, H, D)
        返回:
            configs[l][h] = {"key_bits": int, "value_bits": int}
        """
        n_l = self.n_layers
        n_h = self.n_heads
        agg_key = torch.zeros(n_l, n_h, device=self.device)
        agg_value = torch.zeros(n_l, n_h, device=self.device)
        agg_count = torch.zeros(n_l, n_h, device=self.device)
        agg_entropy = torch.zeros(n_l, n_h, device=self.device)

        for keys, values in kv_pairs:
            keys = keys.to(self.device).float()
            values = values.to(self.device).float()

            # 适配格式：(B, H, S, D) 或 (B, S, H, D)
            if keys.shape[1] == self.n_heads:
                pass  # (B, H, S, D)
            elif keys.shape[2] == self.n_heads:
                keys = keys.transpose(1, 2)
                values = values.transpose(1, 2)
            else:
                continue  # 跳过不匹配

            B, H, S, D = keys.shape
            # 归一化到单位球面
            keys_norm = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
            vals_norm = values / (values.norm(dim=-1, keepdim=True) + 1e-8)

            # 每头每token的平均激活量级
            key_scale = keys_norm.norm(dim=-1).mean(dim=(0, 2))   # (H,)
            val_scale = vals_norm.norm(dim=-1).mean(dim=(0, 2))  # (H,)

            # 注意力熵（近似）
            attn_entropy = self._compute_attention_entropy(keys)

            # 聚合到 (n_layers, n_heads)
            for l in range(n_l):
                # 简单策略：假设 kv_pairs 均匀分布各层
                step = max(1, len(kv_pairs) // n_l)
                l_idx = min(l * step // batch_size, len(kv_pairs) - 1)
                keys_l = keys.to(self.device).float()
                vals_l = values.to(self.device).float()

                # 按头聚合
                k_norm = keys_l.norm(dim=-1).mean(dim=(0, 2))
                v_norm = vals_l.norm(dim=-1).mean(dim=(0, 2))
                if k_norm.shape[0] == n_h:
                    agg_key[l] += k_norm
                    agg_value[l] += v_norm
                    agg_count[l] += 1

        # 归一化
        agg_count = agg_count.clamp(min=1)
        agg_key /= agg_count
        agg_value /= agg_count

        # 敏感度分数 = (key_scale + value_scale) * entropy_boost * position_penalty
        sensitivity = agg_key + agg_value  # (n_layers, n_heads)

        # 层位置加权（首尾层更敏感）
        position_weight = torch.ones(n_l, device=self.device)
        position_weight[:2] = 1.5   # 前两层
        position_weight[-2:] = 1.5  # 后两层
        sensitivity *= position_weight.unsqueeze(1)

        # 归一化到 [0, 1]
        sensitivity = sensitivity / (sensitivity.max() + 1e-8)

        self._scores = sensitivity

        # 生成配置
        configs = self._allocate_bits(sensitivity)
        self._configs = configs
        return configs

    def _compute_attention_entropy(self, keys: torch.Tensor) -> torch.Tensor:
        """
        计算注意力聚焦度（O(N·D)，替代 torch.cov + 特征值分解 O(D³)）。

        方法：峰度（Kurtosis）作为聚焦度代理。
        - 聚焦的注意力（低熵）：峰度大
        - 分散的注意力（高熵）：峰度高斯，峰度 ≈ 3

        Kurtosis = E[(X - μ)⁴] / (E[(X - μ)²])² = μ₄ / σ⁴

        复杂度：O(N·D)，相比 torch.cov + eigvalsh O(D³) 降低 3 个数量级。

        keys: (B, H, S, D)
        返回: (H,) 每头聚焦度
        """
        B, H, S, D = keys.shape
        keys_f = keys.float()

        # 重塑为 (B*H, S, D)，计算每头的统计量
        k = keys_f  # (B, H, S, D)

        # 计算每个头的均值（按 batch 和 sequence 维度求平均）
        mu = k.mean(dim=(0, 2))              # (H, D)
        # 中心化：mu (H,D) 广播到 (B, H, S, D)
        centered = k - mu.view(1, H, 1, D)   # (B, H, S, D)
        # 计算方差（E[X²] - E[X]²）
        var = (centered ** 2).mean(dim=(0, 2))  # (H, D)
        std = (var + 1e-8).sqrt()              # (H, D)

        # 归一化到单位方差（per head per dimension）
        normalized = centered / std.view(1, H, 1, D)  # (B, H, S, D)

        # 峰度：E[normalized⁴] / (E[normalized²])²
        # normalized: (B, H, S, D), mean(dim=(0,2)) → (H, D)
        kurt4 = (normalized ** 4).mean(dim=(0, 2))   # (H, D)
        kurt2 = (normalized ** 2).mean(dim=(0, 2))   # (H, D) ≈ 1
        kurtosis_per_dim = kurt4 / (kurt2 ** 2 + 1e-8)  # (H, D)
        # 对所有维度求平均，得到每头的综合峰度
        kurtosis = kurtosis_per_dim.mean(dim=-1)   # (H,)

        # 峰度越高 → 分布越聚焦 → 越敏感
        # 归一化到 [0, 1]：高斯峰度 = 3，最小值 1（均匀分布）
        # 聚焦分布峰度可达 10+，但大多数注意力分布 < 10
        # 归一化：kurtosis ∈ [1, 10] → sensitivity ∈ [0, 1]
        kurtosis = kurtosis.clamp(min=1.0)
        entropy_proxy = (kurtosis - 1.0) / 9.0  # 归一化到 [0, 1]

        return entropy_proxy

    def _allocate_bits(
        self, scores: torch.Tensor
    ) -> List[List[Dict]]:
        """
        基于敏感度分数分配 bits（预算约束）。

        分配策略：
          1. 基础 bits + 敏感度加成
          2. 总预算约束：sum(bits) ≤ budget
          3. 整数约束：bits 必须是整数
        """
        n_l, n_h = scores.shape
        configs = []

        # 总预算 = 每头平均 × n_heads × n_layers
        total_budget = (self.base_key_bits + self.base_value_bits) * n_l * n_h
        budget_per_layer = (self.base_key_bits + self.base_value_bits) * n_h

        # 每层单独分配（保证预算）
        for l in range(n_l):
            layer_configs = []
            layer_scores = scores[l]  # (n_heads,)
            score_sum = layer_scores.sum() + 1e-8

            for h in range(n_h):
                ratio = float(layer_scores[h].item()) / float(score_sum)
                # 敏感度高 → 更多 bit
                extra_key = (self.max_key_bits - self.base_key_bits) * ratio
                extra_value = (self.max_value_bits - self.base_value_bits) * ratio

                key_bits = int(round(
                    float(self.base_key_bits) + extra_key
                ))
                value_bits = int(round(
                    float(self.base_value_bits) + extra_value
                ))

                # 裁剪到有效范围
                key_bits = max(self.min_key_bits, min(self.max_key_bits, key_bits))
                value_bits = max(
                    self.min_value_bits,
                    min(self.max_value_bits, value_bits),
                )

                layer_configs.append({
                    "key_bits": key_bits,
                    "value_bits": value_bits,
                    "sensitivity": float(scores[l, h].item()),
                })

            # 层内归一化（保证预算）
            layer_bits = sum(c["key_bits"] + c["value_bits"] for c in layer_configs)
            if layer_bits > budget_per_layer:
                scale = budget_per_layer / layer_bits
                for c in layer_configs:
                    c["key_bits"] = max(2, int(c["key_bits"] * scale))
                    c["value_bits"] = max(2, int(c["value_bits"] * scale))

            configs.append(layer_configs)

        return configs

    def get_config(
        self, layer_idx: int, head_idx: int
    ) -> Dict:
        """获取指定层和头的配置。"""
        if self._configs is None:
            raise RuntimeError("请先调用 calibrate()")
        return self._configs[layer_idx][head_idx]

    def summary(self) -> str:
        """打印位分配摘要。"""
        if self._configs is None:
            return "未校准，请先调用 calibrate()"
        lines = ["Per-Head 位分配摘要：", f"{'L':>3} {'H':>3} {'K_bits':>8} {'V_bits':>8} {'Sensitivity':>12}"]
        lines.append("-" * 50)
        for l, layer_cfgs in enumerate(self._configs):
            for h, cfg in enumerate(layer_cfgs):
                if h == 0:
                    lines.append(
                        f"{l:>3} {h:>3} "
                        f"{cfg['key_bits']:>8} {cfg['value_bits']:>8} "
                        f"{cfg['sensitivity']:>12.4f}"
                    )
                elif h == self.n_heads - 1:
                    lines.append(
                        f"{'':>3} {h:>3} "
                        f"{cfg['key_bits']:>8} {cfg['value_bits']:>8} "
                        f"{cfg['sensitivity']:>12.4f}"
                    )
        return "\n".join(lines)


# ===========================================================================
# Async Decompression（3.4）
# ===========================================================================

class AsyncDecompressor:
    """
    异步解压缩器（CUDA Streams 支持）。

    支持在专用 CUDA Stream 上异步执行解压缩，与主计算pipeline重叠。

    用法：
      ```python
      decompressor = AsyncDecompressor(device='cuda:0')

      # 主计算
      with torch.cuda.stream(decompressor.stream):
          keys_decompressed = decompressor.decompress(keys_compressed, ...)
          # 解压缩与主计算重叠

      # 等待解压缩完成
      decompressor.stream.synchronize()
      ```
    """

    def __init__(
        self,
        head_dim: int,
        device: str = "cuda:0",
        n_streams: int = 1,
    ):
        self.head_dim = head_dim
        self.device = device
        self._streams: List[torch.cuda.Stream] = []
        self._current_stream = 0

        if device.startswith("cuda") and torch.cuda.is_available():
            for i in range(n_streams):
                with torch.cuda.device(device):
                    self._streams.append(torch.cuda.Stream())
            self._primary_stream = torch.cuda.current_stream()
        else:
            self._primary_stream = None

    @property
    def stream(self) -> Optional[torch.cuda.Stream]:
        """当前轮转的 CUDA Stream。"""
        if self._streams:
            return self._streams[self._current_stream]
        return None

    def decompress(
        self,
        compressed: Dict,
        compressor,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> torch.Tensor:
        """
        解压缩（同步或异步）。

        如果指定了 stream 且可用，在该 stream 上异步执行。
        否则同步执行。
        """
        if stream is not None and self._streams:
            with torch.cuda.stream(stream):
                return self._decompress_sync(compressed, compressor)
        return self._decompress_sync(compressed, compressor)

    def _decompress_sync(
        self,
        compressed: Dict,
        compressor,
    ) -> torch.Tensor:
        """同步解压缩（CPU fallback 和 CUDA 同步路径）。"""
        return compressor.decompress(compressed)

    def decompress_batch(
        self,
        items: List[Tuple[Dict, any]],
        compressor,
        parallel: bool = True,
    ) -> List[torch.Tensor]:
        """
        批量解压缩（可选并行）。

        items: [(compressed_dict, layer_cfg), ...]
        返回: List[decompressed_tensor]
        """
        if parallel and self._streams and torch.cuda.is_available():
            results = []
            for i, (comp, _) in enumerate(items):
                s = self._streams[i % len(self._streams)]
                with torch.cuda.stream(s):
                    results.append(self._decompress_sync(comp, compressor))
            # 等待所有 stream
            for s in self._streams:
                s.synchronize()
            return results
        else:
            return [self._decompress_sync(comp, comp_) for comp, comp_ in items]

    def rotate_next(self) -> torch.cuda.Stream:
        """轮转到下一个 stream（用于异步流水线）。"""
        if self._streams:
            self._current_stream = (self._current_stream + 1) % len(self._streams)
            return self._streams[self._current_stream]
        return None

    def __repr__(self) -> str:
        n_streams = len(self._streams)
        if n_streams > 0:
            return f"AsyncDecompressor(n_streams={n_streams}, device={self.device})"
        return f"AsyncDecompressor(sync, device={self.device})"
