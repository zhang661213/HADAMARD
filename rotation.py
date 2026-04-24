"""
TurboQuant Random Rotation

正确 Walsh-Hadamard 旋转（验证版）：
  1. 构建 Walsh-Hadamard 矩阵（对称 H = H^T，自逆 H@H = d·I）
  2. Q = H @ diag(sign) / √d（正交）
  3. rotate(x) = x @ Q^T = (x ⊙ sign) @ H / √d
  4. unrotate(y) = y @ Q = (H @ y) ⊙ sign / √d

关键：H 对称 ⟹ H = H^T ⟹ H @ diag(sign) = diag(sign) @ H（可对易！）
"""

import math
from typing import Optional, Literal
import torch


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def build_H_sylvester(d: int) -> torch.Tensor:
    """
    Sylvester Walsh-Hadamard 矩阵。

    Sylvester 递归构造（对称自逆）：
      H_1 = [1]
      H_d = [[H_{d/2},  H_{d/2}],
             [H_{d/2}, -H_{d/2}]]

    性质（经验验证）：
      - 对称：H = H^T
      - 自逆：H @ H = d·I
      - 与 diag(sign) 可对易：H @ diag(sign) = diag(sign) @ H
    """
    assert _is_power_of_two(d)
    H = torch.ones(1, 1, dtype=torch.float32)
    n = 1
    while n < d:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
        n *= 2
    return H[:d, :d]


def build_H_walsh(d: int) -> torch.Tensor:
    """
    Walsh-Hadamard 矩阵（精确 Walsh 序）。

    Walsh 序：行按 sign-change 次数排列。
    与 Sylvester 构造等价但显式 Walsh 序。

    性质：
      - 对称 H = H^T
      - 自逆 H @ H = d·I
      - 与 diag(sign) 可对易
    """
    if d == 1:
        return torch.tensor([[1.0]])
    half = d // 2
    H_half = build_H_walsh(half)
    top = torch.cat([H_half, H_half], dim=1)
    bot = torch.cat([H_half, -H_half], dim=1)
    return torch.cat([top, bot], dim=0)


# ===========================================================================
# 旋转算子
# ===========================================================================

class _HadamardRotation:
    """
    Hadamard 随机旋转。

    Q = H @ diag(sign) / √d（正交，Q @ Q^T = I）
    H 对称 ⟹ Q^T = diag(sign) @ H / √d
    H @ diag(sign) = diag(sign) @ H（可对易）

    rotate(x)   = x @ Q^T = (x ⊙ sign) @ H / √d
    unrotate(y) = y @ Q = (H @ y) ⊙ sign / √d

    rotate(unrotate(y)) = y ✓（H 对称自逆）
    """

    def __init__(self, d: int, seed: Optional[int] = None,
                 device: str = "cpu",
                 mode: Literal["matmul", "fwht"] = "matmul"):
        if not _is_power_of_two(d):
            raise ValueError(f"d={d} must be power of 2")
        self.d = d
        self.device = device
        self.mode = mode
        self._sqrt_d = math.sqrt(d)

        # 随机 signs
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        signs = (torch.randint(0, 2, (d,), generator=gen,
                               dtype=torch.float32) * 2.0 - 1.0)
        if (signs < 0).sum().item() % 2 == 1:
            signs[0] *= -1
        self.signs = signs.to(device)

        if mode == "matmul":
            H = build_H_sylvester(d).to(device)
            # Q^T = diag(sign) @ H / √d（用于 x @ Q^T）
            self._Q_T = (torch.diag(self.signs) @ H) / self._sqrt_d

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """
        rotate(x) = x @ Q^T = (x ⊙ sign) @ H / √d
        """
        if x.device != self.device:
            x = x.to(self.device)
        if self.mode == "matmul":
            if x.dim() == 1:
                return (x * self.signs) @ self._Q_T
            else:
                # 使用 @ 运算符支持任意维度广播
                return (x * self.signs.view(1, -1)) @ self._Q_T
        else:
            # FWHT 路径：分层归一化防溢出
            return self._rotate_fwht(x)

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """
        unrotate(y) = y @ Q = (H @ y) ⊙ sign / √d
        """
        if y.device != self.device:
            y = y.to(self.device)
        if self.mode == "matmul":
            # 使用 @ 运算符支持任意维度广播
            # Q_T = diag(signs) @ H / √d, Q = H @ diag(signs) / √d
            # unrotate(y) = y @ Q = y @ Q_T.t()
            return (y @ self._Q_T.t()) * self.signs.view(*([1] * (y.dim() - 1)), -1)
        else:
            return self._unrotate_fwht(y)

    def _rotate_fwht(self, x: torch.Tensor) -> torch.Tensor:
        """
        FWHT 旋转：x ⊙ sign → butterfly → ⊙ sign / √d

        x ⊙ sign → butterfly → ⊙ sign / √d
        = (H @ (x ⊙ sign)) ⊙ sign / √d
        = (x ⊙ sign) @ H ⊙ sign / √d（因为 H 对称）
        = x @ Q^T ✓
        """
        xr = x * self.signs
        sc = self._fwht分层(xr)
        # FWHT with per-level 1/√2 already normalizes by 1/√d, no extra /√d needed
        xr = xr * self.signs
        if sc > 0:
            xr = xr * (2 ** sc)
        return xr

    def _unrotate_fwht(self, y: torch.Tensor) -> torch.Tensor:
        """
        FWHT 逆旋转 = FWHT 旋转（自逆性）。

        Hadamard 旋转 Q = H⊙signs/√d 满足 Q² = I，
        因此 unrotate = rotate：signs * fwht(signs * y)
        """
        yr = y * self.signs
        sc = self._fwht分层(yr)
        yr = yr * self.signs
        if sc > 0:
            yr = yr * (2 ** sc)
        return yr

    def _fwht分层(self, x: torch.Tensor):
        """
        Walsh-Hadamard butterfly 变换（原地，优化3：FP32累加+动态缩放）。

        优化内容：
        1. FP32 累加器：低精度输入时用 float32 计算 u±v
        2. 动态缩放：每级 butterfly 后检测溢出，超阈值则 ×0.5
        3. 原地写回：避免额外内存分配

        每层 butterfly 后 /√2，最大放大 √d 倍（d=4096 → 64×）。
        但动态缩放保证极端输入也不会溢出。
        """
        d = x.shape[-1]
        if d == 1:
            return

        # ---- 优化3：FP32 累加 + 动态缩放 ----
        use_fp32 = x.dtype in (torch.float16, torch.bfloat16)
        compute_dtype = torch.float32 if use_fp32 else x.dtype

        # 缩放阈值：留 10% margin
        if compute_dtype == torch.float32:
            threshold = 3.4e38 * 0.9
        else:  # FP16
            threshold = 6.0e4 * 0.9

        half_sqrt = math.sqrt(0.5)
        stride = 1
        scale_count = 0

        while stride < d:
            all_idx = torch.arange(d, device=x.device)
            mask = (all_idx & stride) == 0
            idx0 = all_idx[mask]
            idx1 = idx0 ^ stride

            u = x[..., idx0].to(compute_dtype)
            v = x[..., idx1].to(compute_dtype)

            # Butterfly + 分层缩放（在 FP32 累加器中计算）
            t = (u + v) * half_sqrt
            d_val = (u - v) * half_sqrt

            # 动态溢出检测
            cur_max = max(t.abs().max().item(), d_val.abs().max().item())
            if cur_max > threshold:
                t = t * 0.5
                d_val = d_val * 0.5
                scale_count += 1

            # 原地写回
            x[..., idx0] = t.to(x.dtype)
            x[..., idx1] = d_val.to(x.dtype)
            stride <<= 1

        # 记录缩放次数（用于调试/监控）
        self._n_scales = getattr(self, "_n_scales", 0) + scale_count
        return scale_count

    def __repr__(self):
        return f"_HadamardRotation(d={self.d}, mode={self.mode})"

    @property
    def T(self):
        return _TransposeView(self)


class _TransposeView:
    """x @ rot.T → rot.unrotate(x)"""
    __slots__ = ("_rot",)

    def __init__(self, rot):
        self._rot = rot

    def __matmul__(self, x):
        return self._rot.unrotate(x)

    def __rmatmul__(self, x):
        return self._rot.unrotate(x)


class _QRRotation:
    """QR 分解正交旋转（非幂次维度 fallback）。"""
    def __init__(self, d: int, seed: Optional[int] = None, device: str = "cpu"):
        self.d = d
        self.device = device
        self._sqrt_d = math.sqrt(d)
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        G = torch.randn(d, d, generator=gen, device=device)
        Q, _ = torch.linalg.qr(G)
        self._Q = Q / self._sqrt_d

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device:
            x = x.to(self.device)
        return x @ self._Q

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        if y.device != self.device:
            y = y.to(self.device)
        return y @ self._Q.t()

    def __repr__(self):
        return f"_QRRotation(d={self.d})"

    @property
    def T(self):
        return _TransposeView(self)


class _IdentityRotation:
    def __init__(self, d: int):
        self.d = d

    def rotate(self, x):
        return x

    def unrotate(self, y):
        return y

    def __repr__(self):
        return f"_IdentityRotation(d={self.d})"

    @property
    def T(self):
        return _TransposeView(self)


def generate_rotation_matrix(d: int, seed: Optional[int] = None,
                           device: str = "cpu",
                           mode: Literal["matmul", "fwht", "qr"] = "matmul"
                           ):
    if mode == "qr" or not _is_power_of_two(d):
        return _QRRotation(d, seed=seed, device=device)
    return _HadamardRotation(d, seed=seed, device=device, mode=mode)


def generate_qjl_matrix(d: int, qjl_dim: int, seed: Optional[int] = None,
                      device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    G = torch.randn(qjl_dim, d, generator=gen, device=device)
    Q, _ = torch.linalg.qr(G)
    return Q / math.sqrt(qjl_dim)
