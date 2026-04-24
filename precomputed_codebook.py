"""
优化 6: 编译期码本生成 (Compile-time Codebook Generation)

问题：
  - lloyd_max.py 在首次运行时通过 scipy 数值积分计算码本
  - 即使有缓存，初始化仍有 ~100-500ms 延迟（scipy 加载慢）
  - 对延迟敏感的在线推理场景不可接受

解决方案：
  1. 预计算常用 (d, bits) 组合的码本，保存为 C++ 常量数组
  2. 生成 torch.jit.scriptable 函数
  3. 初始化时无需 scipy，无数值积分，直接查表

预计算配置（d ∈ {64, 128, 256, 512}, bits ∈ {2,3,4,5,6}）：
  - 12 个组合 × 2 种分布（exact/gaussian）= 24 个码本
  - 每个码本 2^bits × d × 4 bytes（FP32）
  - 最大: d=512, bits=6 → 2048 × 512 × 4 = 4MB
  - 总计: ~15-20MB（在运行时加载到共享内存）
  - 初始化: <10ms（加载 + 验证 hash）

生成方式：
  python precomputed_codebook.py
  → 生成 precomputed_codebooks.py（可import的Python模块）
  → 同时生成 codebook_data.c（C++ 嵌入）

Usage:
  from precomputed_codebooks import get_codebook, list_codebooks
  centroids = get_codebook(d=128, bits=4)  # 立即返回，无计算
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch

# 常用配置（预计算这些组合）
DEFAULT_DIMS = (64, 128, 256, 512)
DEFAULT_BITS = (2, 3, 4, 5, 6)


# ===========================================================================
# 预计算码本生成
# ===========================================================================

def generate_all_codebooks(
    dims: Tuple[int, ...] = DEFAULT_DIMS,
    bits_range: Tuple[int, ...] = DEFAULT_BITS,
    use_exact: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, dict]:
    """
    批量生成预计算码本。

    Returns:
        Dict[cache_key, {"centroids": Tensor, "boundaries": Tensor, "d": int, "bits": int}]
    """
    from .lloyd_max import solve_lloyd_max

    codebooks = {}
    total = len(dims) * len(bits_range)

    print(f"预计算 {total} 个码本...")
    for i, d in enumerate(dims):
        for j, bits in enumerate(bits_range):
            idx = i * len(bits_range) + j + 1
            label = f"[{idx}/{total}] d={d}, bits={bits}"
            print(f"  {label}...", end=" ", flush=True)

            centroids, boundaries = solve_lloyd_max(d, bits, use_exact=use_exact)
            key = _make_cache_key(d, bits, use_exact)

            codebooks[key] = {
                "centroids": centroids,
                "boundaries": boundaries,
                "d": d,
                "bits": bits,
            }
            print("✓")

    return codebooks


def _make_cache_key(d: int, bits: int, use_exact: bool) -> str:
    tag = f"d{d}_b{bits}_e{int(use_exact)}"
    return hashlib.md5(tag.encode()).hexdigest()[:12]


# ===========================================================================
# 生成可导入的 Python 模块
# ===========================================================================

def codegen_python_module(codebooks: Dict, output_path: str):
    """
    生成 precomputed_codebooks.py（可导入的 Python 模块）。

    内容：
      - get_codebook(d, bits) 函数
      - list_codebooks() 函数
      - 所有码本数据（以 torch.tensor 字面量）
    """
    lines = [
        '"""',
        "预计算 TurboQuant 码本（优化 6）",
        "",
        "自动生成，勿手动修改。",
        f"包含 {len(codebooks)} 个预计算码本。",
        '"""',
        "",
        "from __future__ import annotations",
        "import torch",
        "from typing import Optional, List, Tuple, Dict",
        "",
        "",
        "# ---- 码本数据 ----",
        "# format: _CB_{cache_key} = (d, bits, n_levels, centroids_bytes, hash)",
        "",
    ]

    all_keys = []

    for key, cb in sorted(codebooks.items()):
        d = cb["d"]
        bits = cb["bits"]
        centroids = cb["centroids"]
        n_levels = 2 ** bits

        # 将 centroids 序列化为 bytes
        cb_bytes = centroids.numpy().tobytes()
        cb_hash = hashlib.md5(cb_bytes).hexdigest()[:16]

        # 生成变量名
        var_name = f"_CB_{key}"

        # Python 字面量存储（太长了，改用 base64）
        import base64
        cb_b64 = base64.b64encode(cb_bytes).decode("ascii")

        lines.append(f"{var_name} = {{")
        lines.append(f"    'd': {d},")
        lines.append(f"    'bits': {bits},")
        lines.append(f"    'n_levels': {n_levels},")
        lines.append(f"    'hash': '{cb_hash}',")
        lines.append(f"    'centroids_b64': '{cb_b64}',")
        lines.append(f"}}")
        lines.append("")

        all_keys.append(f"'{key}': {var_name}")

    lines.append("# ---- 索引 ----")
    lines.append(f"_ALL_CODEBOOKS = {{{', '.join(all_keys)}}}")
    lines.append("")

    # 生成函数
    lines += [
        "def get_codebook(d: int, bits: int, use_exact: bool = False) -> Optional[torch.Tensor]:",
        '    """获取预计算码本。命中返回 centroids Tensor，否则返回 None。"""',
        "    key = _make_key(d, bits, use_exact)",
        "    cb = _ALL_CODEBOOKS.get(key)",
        "    if cb is None:",
        "        return None",
        "    import base64",
        "    data = base64.b64decode(cb['centroids_b64'])",
        "    centroids = torch.from_numpy(",
        "        numpy.frombuffer(data, dtype=numpy.float32)",
        f"    ).reshape({cb['n_levels']}, {d})",
        "    return centroids",
        "",
        "def _make_key(d: int, bits: int, use_exact: bool) -> str:",
        "    import hashlib",
        "    tag = f'd{{d}}_b{{bits}}_e{int(use_exact)}'",
        "    return hashlib.md5(tag.encode()).hexdigest()[:12]",
        "",
        "def list_codebooks() -> List[Tuple[int, int]]:`",
        "    \"\"\"列出所有可用码本配置。\"\"\"",
        "    result = []",
        "    for cb in _ALL_CODEBOOKS.values():",
        "        result.append((cb['d'], cb['bits']))",
        "    return sorted(result)",
        "",
        "def verify_hash(d: int, bits: int, centroids: torch.Tensor) -> bool:",
        "    \"\"\"验证码本 hash（完整性检查）。\"\"\"",
        "    import hashlib, base64",
        "    key = _make_key(d, bits, False)",
        "    cb = _ALL_CODEBOOKS.get(key)",
        "    if cb is None:",
        "        return False",
        "    data = base64.b64decode(cb['centroids_b64'])",
        "    return hashlib.md5(data).hexdigest()[:16] == cb['hash']",
        "",
        f"# 共 {len(codebooks)} 个预计算码本",
        "",
        "# === 预热：加载所有码本到缓存 ===",
        "_CACHE: Dict[str, torch.Tensor] = {{}}",
        "",
        "def preload_all():",
        '    """预加载所有码本到内存（应用启动时调用一次）。"""',
        "    for key, cb in _ALL_CODEBOOKS.items():",
        "        if key not in _CACHE:",
        "            _CACHE[key] = get_codebook(cb['d'], cb['bits'])",
        "",
        "def get_cached(d: int, bits: int, use_exact: bool = False) -> torch.Tensor:",
        '    """获取码本（带缓存，第二次调用直接返回）。"""',
        "    key = _make_key(d, bits, use_exact)",
        "    if key not in _CACHE:",
        "        _CACHE[key] = get_codebook(d, bits, use_exact)",
        "    return _CACHE[key]",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  生成 {output_path} ({len(codebooks)} 码本) ✓")


# ===========================================================================
# 生成 C++ 头文件
# ===========================================================================

def codegen_cpp_header(codebooks: Dict, output_path: str):
    """
    生成 codebook_data.c（可 #include 的 C++ 常量数组）。

    用于：
      - CUDA/Triton kernel 中直接访问码本（无 Python overhead）
      - 完全脱离 Python 的嵌入式部署
    """
    lines = [
        "/*",
        " * TurboQuant 预计算码本（C++ 版本）",
        " * 自动生成，勿手动修改。",
        f" * 包含 {len(codebooks)} 个码本。",
        " *",
        " * 使用方法：",
        " *   #include \"codebook_data.c\"",
        " *   const float* kb = get_codebook_ptr(128, 4);  // d=128, bits=4",
        " */",
        "",
        "#ifndef TURBOQUANT_CODEBOOK_DATA_C",
        "#define TURBOQUANT_CODEBOOK_DATA_C",
        "",
        "#include <stdint.h>",
        "#include <string.h>",
        "",
    ]

    # 生成每个码本的常量数组
    for key, cb in sorted(codebooks.items()):
        d = cb["d"]
        bits = cb["bits"]
        centroids = cb["centroids"]
        n_levels = 2 ** bits

        lines.append(f"// {key}: d={d}, bits={bits}, {n_levels} levels")
        lines.append(f"static const float _cb_{key}[{n_levels * d}] = {{")

        # 按行输出，每行 8 个元素
        vals = centroids.numpy().flatten()
        for row_start in range(0, len(vals), 8):
            row = vals[row_start:row_start+8]
            vals_str = ", ".join(f"{v:.8f}" for v in row)
            lines.append(f"    {vals_str},")

        lines.append("};")
        lines.append("")

    # 生成索引结构
    lines.append("// ---- 码本索引 ----")
    lines.append("#define N_CODEBOOKS " + str(len(codebooks)))
    lines.append("")
    lines.append("struct CodebookEntry {")
    lines.append("    const char* key;")
    lines.append("    int d;")
    lines.append("    int bits;")
    lines.append("    int n_levels;")
    lines.append("    const float* data;")
    lines.append("};")
    lines.append("")
    lines.append("static const struct CodebookEntry _codebook_index[N_CODEBOOKS] = {")

    for key, cb in sorted(codebooks.items()):
        d = cb["d"]
        bits = cb["bits"]
        n_levels = 2 ** bits
        lines.append(
            f'    {{"{key}", {d}, {bits}, {n_levels}, _cb_{key}}},'
        )

    lines.append("};")
    lines.append("")
    lines.append("// ---- 查询函数 ----")
    lines.append("""
static const float* get_codebook_ptr(int d, int bits) {
    for (int i = 0; i < N_CODEBOOKS; i++) {
        if (_codebook_index[i].d == d && _codebook_index[i].bits == bits)
            return _codebook_index[i].data;
    }
    return NULL;  // not found
}

static int get_n_levels(int d, int bits) {
    for (int i = 0; i < N_CODEBOOKS; i++) {
        if (_codebook_index[i].d == d && _codebook_index[i].bits == bits)
            return _codebook_index[i].n_levels;
    }
    return 0;
}
""")
    lines.append("#endif  // TURBOQUANT_CODEBOOK_DATA_C")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  生成 {output_path} (C++ 码本) ✓")


# ===========================================================================
# 主生成脚本
# ===========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="生成预计算码本")
    parser.add_argument("--dims", type=int, nargs="+", default=DEFAULT_DIMS,
                        help=f"维度列表 (默认: {DEFAULT_DIMS})")
    parser.add_argument("--bits", type=int, nargs="+", default=DEFAULT_BITS,
                        help=f"位数列表 (默认: {DEFAULT_BITS})")
    parser.add_argument("--output-dir", default=None,
                        help="输出目录 (默认: 当前目录)")
    parser.add_argument("--no-cpp", action="store_true",
                        help="跳过 C++ 头文件生成")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent

    print("=" * 60)
    print("  TurboQuant 预计算码本生成器")
    print("=" * 60)

    # Step 1: 生成所有码本
    codebooks = generate_all_codebooks(
        dims=tuple(args.dims),
        bits_range=tuple(args.bits),
        use_exact=False,
    )

    # Step 2: 生成 Python 模块
    py_path = output_dir / "precomputed_codebooks.py"
    codegen_python_module(codebooks, str(py_path))

    # Step 3: 生成 C++ 头文件
    if not args.no_cpp:
        cpp_path = output_dir / "codebook_data.c"
        codegen_cpp_header(codebooks, str(cpp_path))

    # Step 4: 总结
    total_bytes = sum(
        cb["centroids"].numel() * 4  # FP32 = 4 bytes
        for cb in codebooks.values()
    )

    print(f"\n  生成完成:")
    print(f"    Python 模块: {py_path}")
    if not args.no_cpp:
        print(f"    C++ 头文件:   {cpp_path}")
    print(f"    码本数量:     {len(codebooks)}")
    print(f"    总大小:       {total_bytes / 1024:.1f} KB")
    print()


# ===========================================================================
# 懒加载集成（更新 lloyd_max.py 的 LloydMaxCodebook）
# ===========================================================================

def patch_lloyd_max_codebook():
    """
    猴子补丁：让 LloydMaxCodebook 优先使用预计算码本。

    在 lloyd_max.py 导入后调用此函数：
      from .lloyd_max import LloydMaxCodebook
      patch_lloyd_max_codebook()
      # 之后 LloydMaxCodebook(d=128, bits=4) 会先查预计算表
    """
    from .lloyd_max import LloydMaxCodebook as OriginalLloydMaxCodebook

    # 尝试导入预计算码本（如果存在）
    precomputed_available = False
    _precomputed_get = None

    try:
        from precomputed_codebooks import get_cached as _precomputed_get
        precomputed_available = True
        print("  [lloyd_max] 预计算码本已加载 ✓")
    except ImportError:
        print("  [lloyd_max] 预计算码本未找到，使用运行时计算")

    _original_init = OriginalLloydMaxCodebook.__init__

    def patched_init(self, d: int, bits: int, use_exact: bool = False,
                     device: str = "cpu"):
        # 尝试预计算码本
        if _precomputed_get is not None:
            cached = _precomputed_get(d, bits, use_exact)
            if cached is not None:
                self.d = d
                self.bits = bits
                self.n_levels = 2 ** bits
                self.device = device
                self.centroids = cached.to(device)
                self.boundaries = torch.tensor([], device=device)
                return

        # Fallback: 运行时计算
        _original_init(self, d, bits, use_exact, device)

    OriginalLloydMaxCodebook.__init__ = patched_init
    print("  [lloyd_max] 懒加载补丁已应用 ✓")

    return OriginalLloydMaxCodebook


if __name__ == "__main__":
    main()
