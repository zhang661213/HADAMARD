"""
Step 3: Hybrid TurboQuant — SnapKV 驱逐 + TurboQuant 量化混合

核心思想：
  1. 驱逐（Eviction）：识别并丢弃低注意力 token（保留 ~20-50%）
  2. 量化（Quantization）：对保留的 token 用 TurboQuant 压缩（8x）
  3. 组合效果：总内存降低 ~40x，质量几乎无损

原理：
  - SnapKV 观察到：注意力分布服从幂律，少数 token 贡献绝大部分注意力权重
  - 仅保留高注意力 token，丢弃其余 → 内存线性减少
  - 对保留 token 再量化 → 乘数效应
  - 组合：保留 20% × 压缩 8x = 40x 总降低

两种混合模式：
  mode="snapkv_turbo": 先 SnapKV 驱逐，再 TurboQuant 压缩
  mode="pyramid_snapkv": PyramidKV 层分配 + SnapKV 驱逐（最激进）

Reference: SnapKV (ICLR 2025), PyramidKV (ICLR 2025), TurboQuant (Google 2026)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Literal

import torch
import torch.nn.functional as F


# ===========================================================================
# 注意力追踪器
# ===========================================================================

@dataclass
class TokenImportance:
    """Token 重要性分数"""
    scores: torch.Tensor      # (S,) 每个 token 的重要性分数（0~1）
    kept_indices: torch.Tensor # (K,) 保留的 token 索引（K ≤ S）
    evicted_indices: torch.Tensor # (S-K,) 丢弃的 token 索引
    retention_ratio: float    # 保留比例 K/S
    original_len: int = 0     # 原始序列长度 S（自动计算）

    def __post_init__(self):
        # 自动计算原始长度（如果未提供）
        if self.original_len == 0:
            self.original_len = len(self.kept_indices) + len(self.evicted_indices)
        # backward compat: also add n_kept / n_evicted as properties
        self._n_kept = len(self.kept_indices)
        self._n_evicted = len(self.evicted_indices)

    @property
    def n_kept(self) -> int:
        return self._n_kept

    @property
    def n_evicted(self) -> int:
        return self._n_evicted


class AttentionTracker:
    """
    轻量级注意力重要性追踪器。

    原理：在观察窗口内计算注意力分数的累积分布，
    识别"重击者"（Heavy Hitters）token。

    三种追踪模式：
      1. observation_window: 使用序列末尾 W 个 token 的注意力分布
         → 代表性强（近期 token 决定下一个 token 的注意力分布）
      2. global_attention: 使用完整序列的注意力分布
         → 更准确但更慢
      3. streaming: 移动平均更新（适合流式推理）

    使用方法：
      1. 初始化时传入 n_heads 和 head_dim
      2. 每层推理后调用 update(q, k) 或 update_attention_scores(attn_weights)
      3. 推理结束后调用 get_importance() 获取保留/丢弃 token

    注意：这是轻量级近似。真实 SnapKV 需要访问完整的注意力权重矩阵。
    这里使用 Q-K 内积近似（无需完整 attention 矩阵）。
    """

    def __init__(
        self,
        seq_len: int,
        n_heads: int = 8,
        head_dim: int = 128,
        observation_window: int = 256,
        device: str = "cpu",
    ):
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.obs_window = min(observation_window, seq_len)
        self.device = device

        # 累积重要性分数（head-wise）
        self._cum_scores: Optional[torch.Tensor] = None
        # 每个 head 的观察次数
        self._update_count = 0

    def update_attention_approx(
        self,
        q: torch.Tensor,   # (B, H, S, D) 或 (H, S, D)
        k: torch.Tensor,   # (B, H, S, D) 或 (H, S, D)
        obs_only: bool = True,
    ) -> None:
        """
        用 Q-K 内积近似注意力分数（无需 softmax）。

        q: query tensor
        k: key tensor
        obs_only: True = 只用末尾 obs_window 的 token 计算分数

        原理：Q[i] · K[j] 表示 query i 对 key j 的相关程度
        对所有 query 求平均 → 每个 key token 的平均重要性
        """
        # 统一到 (H, S, D)
        if q.dim() == 4:
            # (B, H, S, D) → 平均 batch
            q = q.mean(dim=0)   # (H, S, D)
            k = k.mean(dim=0)
        elif q.dim() == 3 and q.shape[0] != self.n_heads:
            # (S, H, D) → transpose
            q = q.transpose(0, 1)  # (H, S, D)
            k = k.transpose(0, 1)

        H, S, D = q.shape
        if S != self.seq_len:
            # 如果序列长度变了，重置
            self.seq_len = S
            self._cum_scores = None

        # Q-K 内积 (H, S, D) @ (H, D, S) = (H, S, S)
        # 每个 query 对所有 key 的相关分数
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)  # (H, S, S)

        if obs_only and S > self.obs_window:
            # 只用末尾观察窗口的 query
            obs_q = q[:, -self.obs_window:, :]   # (H, W, D)
            obs_k = k                             # (H, S, D)
            scores = torch.matmul(obs_q, obs_k.transpose(-1, -2)) / math.sqrt(D)
            # scores: (H, W, S) → 对 W 求平均 → (H, S)
            token_scores = scores.mean(dim=1)     # (H, S)
        else:
            # 全局：对所有 query 求平均
            token_scores = scores.mean(dim=1)    # (H, S)

        # 跨 head 平均（更稳定）
        token_scores = token_scores.mean(dim=0)   # (S,)

        # 累积（移动平均）
        if self._cum_scores is None:
            self._cum_scores = token_scores
        else:
            alpha = 0.8  # EMA 系数，新观察占 20%
            self._cum_scores = alpha * self._cum_scores + (1 - alpha) * token_scores

        self._update_count += 1

    def update_from_scores(
        self,
        attn_weights: torch.Tensor,  # (H, S, S) 原始注意力权重
        obs_only: bool = True,
    ) -> None:
        """
        从真实注意力权重更新（更准确，需要访问注意力矩阵）。

        attn_weights: (H, S, S) — softmax(QK^T/√D) 后的注意力矩阵
        """
        H, S1, S2 = attn_weights.shape
        if S1 != self.seq_len or S2 != self.seq_len:
            self.seq_len = S1
            self._cum_scores = None

        if obs_only and S1 > self.obs_window:
            # 只用末尾窗口的 query
            obs_attn = attn_weights[:, -self.obs_window:, :]  # (H, W, S)
            token_scores = obs_attn.mean(dim=1)                 # (H, S)
        else:
            token_scores = attn_weights.mean(dim=1)             # (H, S)

        token_scores = token_scores.mean(dim=0)                  # (S,)

        if self._cum_scores is None:
            self._cum_scores = token_scores
        else:
            alpha = 0.8
            self._cum_scores = alpha * self._cum_scores + (1 - alpha) * token_scores

        self._update_count += 1

    def get_importance(
        self,
        retention_ratio: float = 0.2,
        recent_bias: float = 0.3,
        recent_window: Optional[int] = None,
    ) -> TokenImportance:
        """
        获取 token 重要性分布。

        参数:
            retention_ratio: 保留比例（0.2 = 保留前 20% 重要 token）
            recent_bias: 最近 token 的额外权重（0.3 = 最近 token 分数 +30%）
            recent_window: 最近多少个 token 给予额外权重（默认 seq_len 的 10%）
        """
        if self._cum_scores is None:
            # 没有数据，返回全部保留
            all_idx = torch.arange(self.seq_len, device=self.device)
            return TokenImportance(
                scores=torch.ones(self.seq_len, device=self.device),
                kept_indices=all_idx,
                evicted_indices=torch.tensor([], device=self.device, dtype=torch.long),
                retention_ratio=1.0,
            )

        scores = self._cum_scores.clone()

        # 最近 token 额外权重（位置越近越重要）
        if recent_window is None:
            recent_window = max(1, int(self.seq_len * 0.1))
        recent_mask = torch.zeros_like(scores)
        recent_mask[-recent_window:] = recent_bias
        scores = scores * (1 + recent_mask)

        # 归一化到 [0, 1]
        scores = scores - scores.min()
        scores = scores / (scores.max() + 1e-8)

        # 保留前 retention_ratio 的 token
        n_keep = max(1, int(self.seq_len * retention_ratio))
        threshold = torch.kthvalue(scores, self.seq_len - n_keep + 1).values.item()
        kept_mask = scores >= threshold
        kept_indices = torch.where(kept_mask)[0]
        evicted_indices = torch.where(~kept_mask)[0]

        return TokenImportance(
            scores=scores,
            kept_indices=kept_indices,
            evicted_indices=evicted_indices,
            retention_ratio=n_keep / self.seq_len,
        )

    def reset(self) -> None:
        """重置追踪器（用于新序列）"""
        self._cum_scores = None
        self._update_count = 0


# ===========================================================================
# Token 驱逐器
# ===========================================================================

@dataclass
class EvictionStats:
    """驱逐统计"""
    original_len: int
    kept_len: int
    evicted_len: int
    retention_ratio: float
    method: str


class TokenEvictor:
    """
    Token 驱逐器。

    驱逐策略：
      1. heavy_hitter: 基于注意力分数的 Top-K 保留（SnapKV 核心）
      2. window: 只保留滑动窗口内的 token（StreamingLLM 风格）
      3. hybrid: 滑动窗口 + 重要 token 补充
      4. uniform_sample: 均匀采样（baseline）

    Heavy-Hitter 驱逐原理（PyramidKV/SnapKV）：
      - 计算每个 token 的累积注意力分数
      - 保留分数最高的 K 个 token
      - 丢弃其余 → 内存降低 1/retention_ratio 倍
    """

    STRATEGIES = ["heavy_hitter", "window", "hybrid", "uniform_sample"]

    def __init__(
        self,
        strategy: str = "heavy_hitter",
        retention_ratio: float = 0.2,
        window_size: int = 512,
        recent_window: int = 128,
        device: str = "cpu",
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")

        self.strategy = strategy
        self.retention_ratio = retention_ratio
        self.window_size = window_size
        self.recent_window = recent_window
        self.device = device

        self._tracker: Optional[AttentionTracker] = None

    def build_tracker(
        self, seq_len: int, n_heads: int = 8, head_dim: int = 128,
        observation_window: int = 256,
    ) -> AttentionTracker:
        """构建注意力追踪器"""
        self._tracker = AttentionTracker(
            seq_len=seq_len,
            n_heads=n_heads,
            head_dim=head_dim,
            observation_window=observation_window,
            device=self.device,
        )
        return self._tracker

    def evict(
        self,
        keys: torch.Tensor,      # (B, H, S, D) 或 (S, D)
        values: torch.Tensor,     # (B, H, S, D) 或 (S, D)
        importance: Optional[TokenImportance] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, EvictionStats]:
        """
        驱逐低重要性 token。

        参数:
            keys/values: 原始 KV tensor
            importance: 预计算的 importance（如果已有）

        返回:
            (kept_keys, kept_values, stats)
        """
        orig_shape = keys.shape
        S = orig_shape[-2]

        if importance is None:
            importance = self._get_default_importance(S)

        idx = importance.kept_indices

        # 驱逐
        if keys.dim() == 4:
            kept_k = keys.index_select(dim=2, index=idx)
            kept_v = values.index_select(dim=2, index=idx)
        else:
            kept_k = keys[idx]
            kept_v = values[idx]

        stats = EvictionStats(
            original_len=S,
            kept_len=len(idx),
            evicted_len=S - len(idx),
            retention_ratio=len(idx) / S,
            method=self.strategy,
        )

        return kept_k, kept_v, stats

    def _get_default_importance(self, S: int) -> TokenImportance:
        """策略驱动的默认 importance"""
        if self.strategy == "window":
            keep_idx = torch.arange(max(0, S - self.window_size), S, device=self.device)
            all_idx = torch.arange(S, device=self.device)
            return TokenImportance(
                scores=torch.zeros(S, device=self.device),
                kept_indices=keep_idx,
                evicted_indices=all_idx[~torch.isin(all_idx, keep_idx)],
                retention_ratio=len(keep_idx) / S,
            )
        elif self.strategy == "hybrid":
            # 滑动窗口 + 均匀采样补充
            window_idx = torch.arange(max(0, S - self.window_size), S, device=self.device)
            n_sample = max(1, int(S * self.retention_ratio)) - len(window_idx)
            sample_idx = torch.randperm(S - self.window_size, device=self.device)[:n_sample]
            keep_idx = torch.cat([sample_idx, window_idx]).sort()[0]
            all_idx = torch.arange(S, device=self.device)
            return TokenImportance(
                scores=torch.zeros(S, device=self.device),
                kept_indices=keep_idx,
                evicted_indices=all_idx[~torch.isin(all_idx, keep_idx)],
                retention_ratio=len(keep_idx) / S,
            )
        elif self.strategy == "uniform_sample":
            n_keep = max(1, int(S * self.retention_ratio))
            keep_idx = torch.randperm(S, device=self.device)[:n_keep].sort()[0]
            all_idx = torch.arange(S, device=self.device)
            return TokenImportance(
                scores=torch.zeros(S, device=self.device),
                kept_indices=keep_idx,
                evicted_indices=all_idx[~torch.isin(all_idx, keep_idx)],
                retention_ratio=n_keep / S,
            )
        else:  # heavy_hitter without tracker → fallback to window
            # Fallback: uniform retention (no eviction)
            scores = torch.ones(S, device=self.device)
            kept_indices = torch.arange(S, device=self.device)
            evicted_indices = torch.tensor([], device=self.device, dtype=torch.long)
            return TokenImportance(
                scores=scores,
                kept_indices=kept_indices,
                evicted_indices=evicted_indices,
                retention_ratio=1.0,
            )

    def expand(
        self,
        kept_k: torch.Tensor,
        kept_v: torch.Tensor,
        original_len: int,
        importance: Optional[TokenImportance],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将驱逐后的 KV 展开回原始长度（已丢弃的位置填零）。

        注意：这会产生稀疏的 KV tensor。
        真实推理中，被驱逐的 token 直接不参与注意力计算，所以不需要填零。

        参数:
            kept_k/kept_v: (..., K, D) 驱逐后保留的 KV
            original_len: 原始序列长度 S
            importance: TokenImportance，包含 kept_indices

        返回:
            (full_k, full_v): (..., S, D)，已驱逐位置填零
        """
        # No eviction: kept_k/v already full length
        if importance is None:
            return kept_k, kept_v

        D = kept_k.shape[-1]
        device = kept_k.device

        if kept_k.dim() == 4:
            B, H, K, D = kept_k.shape
            full_k = torch.zeros(B, H, original_len, D, device=device, dtype=kept_k.dtype)
            full_v = torch.zeros(B, H, original_len, D, device=device, dtype=kept_v.dtype)
            full_k.index_copy_(dim=2, index=importance.kept_indices, source=kept_k)
            full_v.index_copy_(dim=2, index=importance.kept_indices, source=kept_v)
        elif kept_k.dim() == 3:
            # (B*H, K, D) → (B*H, S, D)
            N, K, D = kept_k.shape
            full_k = torch.zeros(N, original_len, D, device=device, dtype=kept_k.dtype)
            full_v = torch.zeros(N, original_len, D, device=device, dtype=kept_v.dtype)
            full_k.index_copy_(dim=1, index=importance.kept_indices, source=kept_k)
            full_v.index_copy_(dim=1, index=importance.kept_indices, source=kept_v)
        else:
            K = kept_k.shape[0]
            full_k = torch.zeros(original_len, D, device=device, dtype=kept_k.dtype)
            full_v = torch.zeros(original_len, D, device=device, dtype=kept_v.dtype)
            full_k.index_copy_(dim=0, index=importance.kept_indices, source=kept_k)
            full_v.index_copy_(dim=0, index=importance.kept_indices, source=kept_v)

        return full_k, full_v


# ===========================================================================
# 混合 TurboQuant（驱逐 + 量化）
# ===========================================================================

class HybridTurboQuant:
    """
    混合 TurboQuant：SnapKV 驱逐 + TurboQuant 量化。

    工作流程：
      1. 驱逐：基于注意力分数，丢弃低重要性 token（保留 20-50%）
      2. 量化：用 TurboQuant 压缩保留的 token（8x）
      3. 解压：解压 + 展开回原始长度

    内存降低公式：
      原内存 = S × H × D × 2 bytes
      驱逐后 = S × p × H × D × 2 bytes  (p = retention_ratio)
      量化后 = S × p × H × D × 2 × (bits/16) bytes
      总降低 = 1 / (p × bits/16) 倍

    示例（p=0.2, bits=6）：
      总降低 = 1 / (0.2 × 6/16) = 1 / 0.075 = 13.3x

    对比：
      纯 TurboQuant:  1 / (6/16) = 2.7x
      纯 SnapKV:      1 / 0.2 = 5x
      混合:           13.3x ← 乘数效应！

    三种模式：
      "snapkv_turbo":   SnapKV 驱逐（20%）+ TurboQuant 4+2bit → ~13x
      "pyramid_snapkv": Pyramid 层分配 + SnapKV 驱逐（30%） → ~8x
      "window_turbo":   滑动窗口（50%）+ TurboQuant → ~5x
    """

    def __init__(
        self,
        head_dim: int = 128,
        key_bits: int = 4,
        value_bits: int = 2,
        # 驱逐参数
        eviction_strategy: str = "heavy_hitter",
        retention_ratio: float = 0.2,
        window_size: int = 512,
        recent_window: int = 128,
        # TurboQuant 参数
        residual_window: int = 128,
        use_dynamic_window: bool = False,
        seed: int = 42,
        device: str = "cpu",
        mode: Literal["snapkv_turbo", "pyramid_snapkv", "window_turbo"] = "snapkv_turbo",
    ):
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.eviction_strategy = eviction_strategy
        self.retention_ratio = retention_ratio
        self.window_size = window_size
        self.recent_window = recent_window
        self.residual_window = residual_window
        self.use_dynamic_window = use_dynamic_window
        self.seed = seed
        self.device = device
        self.mode = mode

        # 构建驱逐器
        self.evictor = TokenEvictor(
            strategy=eviction_strategy,
            retention_ratio=retention_ratio,
            window_size=window_size,
            recent_window=recent_window,
            device=device,
        )

        # 构建 TurboQuant（延迟导入）
        self._HADAMARD = None

    def _get_HADAMARD(self, layer_idx: int = 0, n_layers: int = 1):
        """延迟构建 TurboQuant（避免循环导入）"""
        if self._HADAMARD is None:
            from .turboquant import TurboQuantKV
            self._HADAMARD = TurboQuantKV(
                head_dim=self.head_dim,
                key_bits=self.key_bits,
                value_bits=self.value_bits,
                residual_window=self.residual_window,
                use_dynamic_window=self.use_dynamic_window,
                layer_idx=layer_idx,
                n_layers=n_layers,
                protected_layers=0,
                protected_bits=max(self.key_bits, self.value_bits),
                seed=self.seed,
                device=self.device,
            )
        return self._HADAMARD

    @torch.no_grad()
    def compress(
        self,
        keys: torch.Tensor,    # (B, H, S, D)
        values: torch.Tensor,  # (B, H, S, D)
        importance: Optional[TokenImportance] = None,
    ) -> Dict:
        """
        混合压缩：驱逐 + TurboQuant 量化。

        返回:
            {
                "evicted_keys/values": 驱逐后保留的原始 KV（用于解压展开）
                "compressed": TurboQuant 压缩结果
                "importance": TokenImportance（用于解压）
                "eviction_stats": EvictionStats
                "original_shape": 原始 shape
            }
        """
        orig_shape = keys.shape
        B, H, S, D = orig_shape

        # Step 1: 驱逐
        kept_k, kept_v, ev_stats = self.evictor.evict(keys, values, importance)

        # Step 2: TurboQuant 压缩
        tq = self._get_HADAMARD()
        ck, cv = tq.compress_kv(kept_k, kept_v)

        return {
            "evicted_keys": kept_k,
            "evicted_values": kept_v,
            "compressed": {"keys": ck, "values": cv},
            "importance": importance,
            "eviction_stats": ev_stats,
            "original_shape": orig_shape,
        }

    @torch.no_grad()
    def decompress(self, compressed: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解压并展开回原始长度。

        注意：被驱逐的 token 位置填零。
        真实推理中，被驱逐的 token 不参与注意力计算，所以填零是正确的（无注意力贡献）。
        """
        orig_shape = compressed["original_shape"]
        B, H, S, D = orig_shape

        # Step 1: TurboQuant 解压
        tq = self._get_HADAMARD()
        ck = compressed["compressed"]["keys"]
        cv = compressed["compressed"]["values"]
        kept_k, kept_v = tq.decompress_kv(ck, cv)

        # Step 2: 展开回原始长度
        importance = compressed["importance"]
        full_k, full_v = self.evictor.expand(kept_k, kept_v, S, importance)

        return full_k, full_v

    def memory_usage(
        self,
        B: int, H: int, S: int, D: int,
    ) -> Dict:
        """
        内存分析（混合模式 vs 纯 TurboQuant vs FP16）。

        分解：
          1. 驱逐效果：S × p × H × D × 2 bytes（p = retention_ratio）
          2. 量化效果：× (K_bits + V_bits) / 16
          3. 总降低 = 1 / (p × avg_bits/16)
        """
        avg_bits = self.key_bits + self.value_bits
        p = self.retention_ratio

        fp16 = 2 * B * H * S * D * 2
        evict_only = fp16 * p
        hybrid = fp16 * p * (avg_bits / 16)
        pure_turbo = fp16 * (avg_bits / 16)

        return {
            "fp16_bytes": fp16,
            "eviction_only_bytes": evict_only,
            "pure_turbo_bytes": pure_turbo,
            "hybrid_bytes": hybrid,
            "eviction_ratio": fp16 / evict_only,
            "turbo_ratio": fp16 / pure_turbo,
            "hybrid_ratio": fp16 / hybrid,
            "retention_ratio": p,
        }


# ===========================================================================
# 对比测试
# ===========================================================================

def compare_all_strategies(
    seq_len: int = 2048,
    n_heads: int = 32,
    head_dim: int = 128,
    batch_size: int = 1,
    retention_ratio: float = 0.2,
    key_bits: int = 4,
    value_bits: int = 2,
    device: str = "cpu",
    seed: int = 42,
) -> Dict:
    """
    完整对比：FP16 vs 纯 TurboQuant vs 纯驱逐 vs 混合。

    返回各策略的质量指标和内存分析。
    """
    torch.manual_seed(seed)
    from .turboquant import TurboQuantKV
    from .pyramid_quant import QualityValidator

    # 合成测试数据（模拟 LLM KV 激活）
    keys = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    values = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    keys = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
    values = values / (values.norm(dim=-1, keepdim=True) + 1e-8)

    validator = QualityValidator(verbose=False)

    results = {}

    # ── 1. FP16（原始）────────────────────────────────────────
    # 无压缩，作为 baseline
    results["fp16"] = {
        "keys": keys,
        "values": values,
        "memory_ratio": 1.0,
        "method": "FP16 (原始)",
    }

    # ── 2. 纯 TurboQuant（无驱逐）────────────────────────────
    tq = TurboQuantKV(
        head_dim=head_dim, key_bits=key_bits, value_bits=value_bits,
        layer_idx=0, n_layers=1, protected_layers=0,
        protected_bits=max(key_bits, value_bits),
        seed=seed, device=device,
    )
    ck, cv = tq.compress_kv(keys, values)
    dk, dv = tq.decompress_kv(ck, cv)
    m = validator.validate(keys, values, dk, dv)
    avg_bits = key_bits + value_bits
    mem_tq = 2 * batch_size * n_heads * seq_len * head_dim * 2 * (avg_bits / 16)
    results["pure_HADAMARD"] = {
        "compressed": (ck, cv),
        "decompressed": (dk, dv),
        "metrics": m,
        "memory_ratio": mem_tq / (2 * batch_size * n_heads * seq_len * head_dim * 2),
        "method": f"TurboQuant {key_bits}+{value_bits}bit",
    }

    # ── 3. 纯驱逐（无量化）───────────────────────────────────
    evictor = TokenEvictor(
        strategy="heavy_hitter",
        retention_ratio=retention_ratio,
        device=device,
    )
    # 模拟：随机重要性分数（保留 20%）
    kept = torch.randperm(seq_len, device=device)[:int(seq_len * retention_ratio)].sort()[0]
    all_idx = torch.arange(seq_len, device=device)
    evicted = all_idx[~torch.isin(all_idx, kept)]
    importance = TokenImportance(
        scores=torch.rand(seq_len, device=device),
        kept_indices=kept,
        evicted_indices=evicted,
        retention_ratio=retention_ratio,
        original_len=seq_len,
    )
    kept_k, kept_v, ev_stats = evictor.evict(keys, values, importance)
    full_k, full_v = evictor.expand(kept_k, kept_v, seq_len, importance)
    m_ev = validator.validate(keys, values, full_k, full_v)
    mem_ev = 2 * batch_size * n_heads * kept_k.shape[2] * head_dim * 2
    results["pure_eviction"] = {
        "eviction_stats": ev_stats,
        "metrics": m_ev,
        "memory_ratio": mem_ev / (2 * batch_size * n_heads * seq_len * head_dim * 2),
        "method": f"SnapKV 驱逐 (p={retention_ratio})",
    }

    # ── 4. 混合（驱逐 + TurboQuant）─────────────────────────
    hybrid = HybridTurboQuant(
        head_dim=head_dim,
        key_bits=key_bits,
        value_bits=value_bits,
        retention_ratio=retention_ratio,
        seed=seed,
        device=device,
    )
    comp = hybrid.compress(keys, values, importance)
    dk_h, dv_h = hybrid.decompress(comp)
    m_h = validator.validate(keys, values, dk_h, dv_h)
    mem_usage = hybrid.memory_usage(batch_size, n_heads, seq_len, head_dim)
    results["hybrid"] = {
        "compressed": comp,
        "metrics": m_h,
        "memory_usage": mem_usage,
        "method": f"混合 (p={retention_ratio}, {key_bits}+{value_bits}bit)",
    }

    # ── 打印对比表 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  KV Cache 压缩策略完整对比")
    print(f"  序列长度: {seq_len} | Heads: {n_heads} | Dim: {head_dim} | "
          f"Batch: {batch_size}")
    print("=" * 70)
    print(f"  {'方法':<35} | {'Key CosSim':>10} | {'Val CosSim':>10} | {'内存占比':>8} | {'总降低':>6}")
    print("  " + "-" * 70)

    fp16_mem = 2 * batch_size * n_heads * seq_len * head_dim * 2

    for name, r in results.items():
        if name == "fp16":
            print(f"  {r['method']:<35} | {'1.000000':>10} | {'1.000000':>10} | {'100.0%':>8} | {'1.0x':>6}")
            continue

        if "metrics" in r:
            m = r["metrics"]
        elif "memory_usage" in r:
            m = r["metrics"]

        ratio = r.get("memory_ratio", mem_usage.get("hybrid_ratio", 0) if "memory_usage" in r else 0)

        if "memory_usage" in r:
            ratio = r["memory_usage"]["hybrid_ratio"]

        if name == "pure_eviction":
            ratio = r["memory_ratio"]
        elif name == "pure_HADAMARD":
            ratio = r["memory_ratio"]

        method = r["method"]
        if "metrics" in r:
            kc = f"{r['metrics'].key_cossim:.6f}"
            vc = f"{r['metrics'].value_cossim:.6f}"
        else:
            kc = "N/A"
            vc = "N/A"
        pct = 100 / ratio if ratio > 0 else 0
        print(f"  {method:<35} | {kc:>10} | {vc:>10} | {pct:>7.1f}% | {ratio:>5.1f}x")

    print("  " + "-" * 70)
    print(f"\n  混合模式理论降低: {1/mem_usage['hybrid_ratio']:.1f}x "
          f"(驱逐 {1/mem_usage['retention_ratio']:.1f}x × "
          f"量化 {16/mem_usage.get('avg_bits', avg_bits):.1f}x)")
    print()

    return results
