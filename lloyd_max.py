"""
TurboQuant Lloyd-Max Optimal Scalar Quantizer

优化 5：持久化码本存储。
  - 预计算常用 (d, bits) 组合的最优质心到 .pt 缓存
  - 初始化时先查缓存，命中则直接加载（毫秒级）
  - 首次计算后自动缓存到磁盘
  - 预计算工具 precompute_all() 批量生成缓存
"""

import math
import os
import hashlib
from pathlib import Path
from typing import Optional

import torch

# Lazy import scipy only when needed (scipy is slow to load ~2-5s)
_integrate = None

def _get_integrate():
    global _integrate
    if _integrate is None:
        from scipy import integrate as _int
        _integrate = _int
    return _integrate


# ===========================================================================
# 码本缓存管理器
# ===========================================================================

class CodebookCache:
    """
    Lloyd-Max 码本持久化缓存。

    缓存结构：
      cache_dir/
        lloyd_max_d064_b02.pt   # d=64, bits=2
        lloyd_max_d064_b04.pt   # d=64, bits=4
        ...
        index.json               # 缓存索引（可选）

    缓存键：md5(f"d={d}_bits={bits}_exact={use_exact}")
    """
    DEFAULT_CACHE_DIR = Path(__file__).parent / ".codebook_cache"

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, d: int, bits: int, use_exact: bool) -> str:
        """生成缓存文件名。"""
        tag = f"d{d}_b{bits}_e{int(use_exact)}"
        return f"lloyd_max_{hashlib.md5(tag.encode()).hexdigest()[:12]}"

    def load(self, d: int, bits: int, use_exact: bool
             ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """
        尝试从缓存加载。
        返回 (centroids, boundaries) 或 None。
        """
        key = self._key(d, bits, use_exact)
        path = self.cache_dir / f"{key}.pt"
        if path.exists():
            try:
                data = torch.load(path, map_location="cpu", weights_only=True)
                # 验证数据完整性
                if (isinstance(data, dict) and
                        "centroids" in data and "boundaries" in data):
                    return data["centroids"], data["boundaries"]
                elif isinstance(data, tuple) and len(data) == 2:
                    return data[0], data[1]
            except Exception:
                pass  # 缓存损坏，忽略
        return None

    def save(self, d: int, bits: int, use_exact: bool,
             centroids: torch.Tensor, boundaries: torch.Tensor):
        """保存码本到缓存。"""
        key = self._key(d, bits, use_exact)
        path = self.cache_dir / f"{key}.pt"
        torch.save(
            {"centroids": centroids, "boundaries": boundaries,
             "d": d, "bits": bits, "use_exact": use_exact,
             "version": 1},
            path
        )

    def clear(self):
        """清空所有缓存。"""
        for p in self.cache_dir.glob("lloyd_max_*.pt"):
            p.unlink()

    def list_cached(self):
        """列出所有缓存文件。"""
        return sorted(self.cache_dir.glob("lloyd_max_*.pt"))

    def __repr__(self):
        cached = len(self.list_cached())
        return f"CodebookCache(dir={self.cache_dir}, cached={cached})"


# ===========================================================================
# Lloyd-Max 求解器
# ===========================================================================

def beta_pdf(x: float, d: int) -> float:
    """精确 Beta 分布 PDF（坐标旋转后的分布）。"""
    if abs(x) >= 1.0:
        return 0.0
    coeff = math.gamma(d / 2) / (math.sqrt(math.pi) * math.gamma((d - 1) / 2))
    return coeff * (1.0 - x * x) ** ((d - 3) / 2)


def gaussian_approx_pdf(x: float, d: int) -> float:
    """高斯近似 N(0, 1/d)。"""
    sigma2 = 1.0 / d
    return (1.0 / math.sqrt(2 * math.pi * sigma2)) * math.exp(-x * x / (2 * sigma2))


def solve_lloyd_max(d: int, bits: int, use_exact: bool = False,
                    max_iter: int = 500, tol: float = 1e-12):
    """
    求解 Lloyd-Max 最优量化器（带缓存）。

    算法：迭代交替执行两步直到收敛
      1. 最近邻划分：边界 = 相邻质心中点
      2. 质心更新：条件期望 E[X|在段内]

    参数:
        d:           向量维度
        bits:        量化位数
        use_exact:   True=精确Beta PDF，False=高斯近似（d≥64时足够好）
        max_iter:    最大迭代次数
        tol:         收敛阈值

    返回:
        centroids:  排序后的 2^bits 个质心 (tensor, float32)
        boundaries: 相邻质心之间的边界 (tensor, float32)
    """
    # 先查缓存
    cache = CodebookCache()
    cached = cache.load(d, bits, use_exact)
    if cached is not None:
        return cached

    # 数值积分求解
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)
    pdf = (lambda x: beta_pdf(x, d)) if use_exact else (lambda x: gaussian_approx_pdf(x, d))

    # 初始化：均匀分布在 [-3.5σ, 3.5σ]
    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]

        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            integrate = _get_integrate()
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            if den > 1e-15:
                new_centroids.append(num / den)
            else:
                new_centroids.append(centroids[i])

        max_shift = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_shift < tol:
            break

    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
    centroids_t = torch.tensor(centroids, dtype=torch.float32)
    boundaries_t = torch.tensor(boundaries, dtype=torch.float32)

    # 自动缓存
    cache.save(d, bits, use_exact, centroids_t, boundaries_t)
    return centroids_t, boundaries_t


# ===========================================================================
# 预计算工具（批量生成缓存）
# ===========================================================================

def precompute_all(
    dims: tuple[int, ...] = (64, 128, 256),
    bits_range: tuple[int, ...] = (2, 3, 4, 5, 6),
    use_exact: bool = False,
    cache_dir: Optional[str] = None,
) -> CodebookCache:
    """
    批量预计算常用 (d, bits) 组合的码本并缓存。

    调用示例：
      cache = precompute_all()  # 首次运行耗时 ~5-10s
      # 后续运行毫秒级加载
    """
    cache = CodebookCache(cache_dir)
    computed = []
    total = len(dims) * len(bits_range)

    print(f"预计算 {total} 个码本到 {cache.cache_dir} ...")
    for i, d in enumerate(dims):
        for bits in bits_range:
            label = f"[{i*len(bits_range)+bits_range.index(bits)+1}/{total}]"
            cached = cache.load(d, bits, use_exact)
            if cached is not None:
                print(f"  {label} d={d}, bits={bits}: CACHE HIT")
            else:
                print(f"  {label} d={d}, bits={bits}: computing...", end=" ", flush=True)
                solve_lloyd_max(d, bits, use_exact)
                print("saved ✓")
                computed.append((d, bits))
    if computed:
        print(f"  新增 {len(computed)} 个缓存文件")
    else:
        print(f"  全部命中缓存")
    return cache


# ===========================================================================
# Lloyd-Max 码本
# ===========================================================================

class LloydMaxCodebook:
    """
    预计算的 Lloyd-Max 码本。

    优化 5：初始化时自动查缓存，命中则毫秒级加载。
    """

    _shared_cache: Optional[CodebookCache] = None

    @classmethod
    def set_cache_dir(cls, cache_dir: str):
        """设置全局缓存目录。"""
        cls._shared_cache = CodebookCache(cache_dir)

    @classmethod
    def preload(cls, dims: tuple[int, ...] = (64, 128, 256),
                bits: int = 4):
        """
        预加载指定维度的码本到缓存（应用启动时调用一次）。
        将触发批量计算，后续所有 LloydMaxCodebook 实例都命中缓存。
        """
        for d in dims:
            solve_lloyd_max(d, bits)

    def __init__(self, d: int, bits: int, use_exact: bool = False,
                 device: str = "cpu"):
        self.d = d
        self.bits = bits
        self.n_levels = 2 ** bits
        self.device = device
        self.centroids, self.boundaries = solve_lloyd_max(d, bits, use_exact)
        self.centroids = self.centroids.to(device)
        self.boundaries = self.boundaries.to(device)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """量化：找最近质心，返回索引。"""
        diffs = x.unsqueeze(-1) - self.centroids.to(x.device)
        return diffs.abs().argmin(dim=-1).to(torch.uint8)

    def dequantize(self, indices: torch.Tensor) -> torch.Tensor:
        """反量化：索引 → 质心值。"""
        return self.centroids.to(indices.device)[indices.long()]

    def __repr__(self):
        return (f"LloydMaxCodebook(d={self.d}, bits={self.bits}, "
                f"levels={self.n_levels})")
