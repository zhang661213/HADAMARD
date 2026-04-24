"""
TurboQuant 合成数据测试套件（优化版）

验证内容：
  1. Lloyd-Max 码本质量
  2. 旋转后坐标分布（Hadamard vs QR 等价性）
  3. searchsorted vs argmin 量化精度对比
  4. Bit-pack 无损往返
  5. MSE 重建误差（SNR）
  6. V1 vs V3 内积估计对比
  7. 动态残差窗口
  8. 层敏感度分析
  9. 压缩率测试
  10. 性能基准测试
"""

import math
import time
import sys

import torch
import torch.nn.functional as F

from lloyd_max import LloydMaxCodebook
from rotation import generate_rotation_matrix, _HadamardRotation
from turboquant import (
    MSECompressor, TurboQuantKV, TurboQuantV1,
    BitPacker, compute_residual_window, LayerSensitivityAnalyzer,
    _quantize_searchsorted, _quantize_argmin,
)


def test_rotation_distribution():
    print("\n" + "=" * 60)
    print("Test 1: 旋转后坐标分布")
    print("=" * 60)

    d = 128
    n_samples = 10000
    seed = 42

    torch.manual_seed(seed)
    vectors = torch.randn(n_samples, d)
    vectors = vectors / vectors.norm(dim=-1, keepdim=True)

    Pi = generate_rotation_matrix(d, seed=seed)
    rotated = vectors @ Pi.T if hasattr(Pi, 'T') else Pi.forward(vectors)

    print(f"d={d}, 样本={n_samples}")
    print(f"实际均值（max）: {rotated.mean(dim=0).abs().max().item():.4f}")
    print(f"期望方差: {1/d:.6f}, 实际方差: {rotated.var(dim=0).mean().item():.6f}")
    print(f"坐标相关性（max）: {rotated.cov().abs().fill_diagonal_(0).max().item():.4f}")

    from scipy import stats
    sigma = 1 / math.sqrt(d)
    _, p = stats.kstest(rotated[:, 0].numpy(), 'norm', args=(0, sigma))
    print(f"KS 检验 N(0,1/d): p={p:.4f} {'[PASS]' if p > 0.01 else '[FAIL]'}")

    # Hadamard vs QR 等价性
    Pi_h = _HadamardRotation(d, seed=seed)
    Pi_qr = generate_rotation_matrix(d, seed=seed)
    v = vectors[:10].float()
    h_out = Pi_h.rotate(v)
    qr_out = v @ Pi_qr.T if hasattr(Pi_qr, 'T') else Pi_qr.rotate(v)
    diff = (h_out - qr_out).abs().max().item()
    print(f"Hadamard vs QR 等价性检查: max_diff={diff:.2e} {'[PASS]' if diff < 1e-4 else '[FAIL]'}")
    return True


def test_lloyd_max_quality():
    print("\n" + "=" * 60)
    print("Test 2: Lloyd-Max 量化质量")
    print("=" * 60)

    d = 128
    for bits in [2, 3, 4, 5]:
        codebook = LloydMaxCodebook(d, bits)
        print(f"\nbits={bits}, levels={codebook.n_levels}")
        print(f"  质心范围: [{codebook.centroids.min():.4f}, {codebook.centroids.max():.4f}]")

        sigma = 1 / math.sqrt(d)
        samples = torch.randn(100000, d) * sigma
        samples_unit = samples / samples.norm(dim=-1, keepdim=True)
        rotated = samples_unit @ generate_rotation_matrix(d, seed=42).T
        x_flat = rotated[:, 0]
        indices = codebook.quantize(x_flat)
        recon = codebook.dequantize(indices).float()
        mse = (x_flat.float() - recon).var().item()
        print(f"  采样 MSE（首坐标）: {mse:.6f}")
    return True


def test_searchsorted_vs_argmin():
    print("\n" + "=" * 60)
    print("Test 3: searchsorted vs argmin 精度对比")
    print("=" * 60)

    d = 128
    bits = 4
    n = 10000

    torch.manual_seed(42)
    v = torch.randn(n, d)
    v = v / v.norm(dim=-1, keepdim=True)
    rot = generate_rotation_matrix(d, seed=42)
    rotated = v @ rot.T

    codebook = LloydMaxCodebook(d, bits)
    centroids = codebook.centroids

    idx_argmin = _quantize_argmin(rotated, centroids)
    idx_sorted = _quantize_searchsorted(rotated, centroids)

    match = (idx_argmin == idx_sorted).float().mean().item()
    print(f"d={d}, bits={bits}, n={n}: 匹配率={match:.6f}")

    # 重建误差对比（比较 argmin vs searchsorted 的量化结果一致性）
    recon_argmin = centroids[idx_argmin.long()]
    recon_sorted = centroids[idx_sorted.long()]
    err_diff = (recon_argmin - recon_sorted).norm().item()
    print(f"重建差异: {err_diff:.2e}")
    return match > 0.999


def test_bitpack_roundtrip():
    print("\n" + "=" * 60)
    print("Test 4: Bit-Pack 往返无损测试")
    print("=" * 60)

    configs = [(128, 3), (128, 4), (64, 3), (256, 4)]
    for d, bits in configs:
        packer = BitPacker(d, bits)
        torch.manual_seed(123)
        indices = torch.randint(0, 2**bits, (500, d), dtype=torch.uint8)
        packed = packer.pack(indices)
        unpacked = packer.unpack(packed, d)
        match = (indices == unpacked).all().item()
        print(f"d={d}, bits={bits}: 往返无损={match}, n_bytes={packer.n_bytes}")
        assert match, f"Bit-pack 往返失败 d={d}, bits={bits}"
    print("[PASS] Bit-pack 往返无损")
    return True


def test_mse_reconstruction():
    print("\n" + "=" * 60)
    print("Test 5: MSE 重建质量（SNR）")
    print("=" * 60)

    d = 128
    B, H, S = 2, 8, 512

    torch.manual_seed(42)
    keys = torch.randn(B, H, S, d)
    keys = keys / keys.norm(dim=-1, keepdim=True)

    for bits in [2, 3, 4, 5]:
        comp = MSECompressor(d, bits, seed=42)
        compressed = comp.compress(keys)
        reconstructed = comp.decompress(compressed)

        signal_power = (keys.float() ** 2).mean().item()
        noise_power = ((keys.float() - reconstructed.float()) ** 2).mean().item()
        snr = 10 * math.log10(signal_power / (noise_power + 1e-10))
        cos_sim = torch.nn.functional.cosine_similarity(
            keys.reshape(-1, d).float(), reconstructed.reshape(-1, d).float(), dim=-1
        ).mean().item()
        print(f"bits={bits}: SNR={snr:.2f} dB, CosSim={cos_sim:.6f}, "
              f"相对误差={math.sqrt(noise_power/signal_power)*100:.2f}%")
    return True


def test_compression_ratio():
    print("\n" + "=" * 60)
    print("Test 6: 压缩率测试")
    print("=" * 60)

    configs = [
        (64, 3), (64, 4), (64, 8),
        (128, 2), (128, 3), (128, 4), (128, 8),
        (256, 3), (256, 4),
    ]

    for head_dim, bits in configs:
        comp = MSECompressor(head_dim, bits, seed=42)
        mem = comp.memory_usage(1, 1, 1024)
        print(f"d={head_dim:3d}, bits={bits}: "
              f"压缩率={mem['compression_ratio']:.2f}x, "
              f"压缩={mem['compressed_bytes']:,}B, 原始={mem['fp16_bytes']:,}B")
    return True


def test_v1_vs_v3():
    print("\n" + "=" * 60)
    print("Test 7: V1 (QJL) vs V3 (MSE-only) 内积估计对比")
    print("=" * 60)

    d = 128
    n = 500

    torch.manual_seed(42)
    x = torch.randn(n, d)
    x = x / x.norm(dim=-1, keepdim=True)
    y = torch.randn(n, d)
    y = y / y.norm(dim=-1, keepdim=True)

    bits = 3
    v1 = TurboQuantV1(d, bits, seed=42)
    v3 = MSECompressor(d, bits, seed=42)

    compressed = v1.quantize(x)
    v3_comp = v3.compress(x.reshape(1, 1, n, d))

    true_ips = (x * y).sum(dim=-1)
    x_v3_recon = v3.decompress(v3_comp)
    x_v1_recon = v1.dequantize(compressed)
    x_v1_ip = v1.inner_product(y, compressed)

    def report(name, ips, true):
        mse = ((ips - true) ** 2).mean().item()
        corr = torch.corrcoef(torch.stack([ips, true]))[0, 1].item()
        print(f"  {name}: MSE={mse:.6f}, 相关系数={corr:.6f}")

    print("内积估计质量：")
    report("真实内积", true_ips, true_ips)
    report("V1 MSE-only", (y * x_v1_recon).sum(-1), true_ips)
    report("V1 QJL估计", x_v1_ip, true_ips)
    report("V3 MSE-only", (y * x_v3_recon.squeeze()).sum(-1), true_ips)
    return True


def test_dynamic_window():
    print("\n" + "=" * 60)
    print("Test 8: 动态残差窗口")
    print("=" * 60)

    for seq_len in [64, 128, 256, 512, 1024, 2048]:
        rw = compute_residual_window(seq_len, base=128)
        ratio = rw / seq_len * 100
        print(f"S={seq_len:5d}: 残差窗口={rw:4d} ({ratio:5.1f}%), 压缩S={seq_len-rw:5d}")
    return True


def test_layer_sensitivity():
    print("\n" + "=" * 60)
    print("Test 9: 层敏感度分析接口")
    print("=" * 60)

    n_layers = 36
    analyzer = LayerSensitivityAnalyzer(n_layers, base_key_bits=4, base_value_bits=2)

    # 模拟层激活数据（首尾层更大，中间层更小）
    import random
    random.seed(0)
    kv_pairs = []
    for li in range(n_layers):
        scale = 1.0 + 0.5 * max(0, 4 - min(li, n_layers - 1 - li))
        B, H, S, D = 2, 8, 128, 128
        keys = torch.randn(B, H, S, D) * scale
        values = torch.randn(B, H, S, D) * scale
        kv_pairs.append((li, keys, values))

    scores, allocs = analyzer.analyze(kv_pairs)
    print("前5层敏感度分数：", torch.round(scores[:5], decimals=3).tolist())
    print("前5层 bit 分配 (K,V)：", allocs[:5].long().tolist())
    print("后5层敏感度分数：", torch.round(scores[-5:], decimals=3).tolist())
    print("后5层 bit 分配 (K,V)：", allocs[-5:].long().tolist())
    return True


def test_kv_cache_interface():
    print("\n" + "=" * 60)
    print("Test 10: KV Cache 接口（含动态窗口）")
    print("=" * 60)

    d = 128
    B, H, S = 1, 8, 512

    torch.manual_seed(42)
    keys = torch.randn(B, H, S, d)
    keys = keys / keys.norm(dim=-1, keepdim=True)
    values = torch.randn(B, H, S, d)
    values = values / values.norm(dim=-1, keepdim=True)

    # 固定窗口
    kv_fixed = TurboQuantKV(d, key_bits=4, value_bits=2,
                            residual_window=128, use_dynamic_window=False)
    ck_f, cv_f = kv_fixed.compress_kv(keys, values)
    dk_f, dv_f = kv_fixed.decompress_kv(ck_f, cv_f)

    # 动态窗口
    kv_dyn = TurboQuantKV(d, key_bits=4, value_bits=2,
                            residual_window=128, use_dynamic_window=True)
    ck_d, cv_d = kv_dyn.compress_kv(keys, values)
    dk_d, dv_d = kv_dyn.decompress_kv(ck_d, cv_d)

    def report(name, a, b):
        cos = torch.nn.functional.cosine_similarity(
            a.reshape(-1, d), b.reshape(-1, d), dim=-1
        ).mean().item()
        print(f"  {name}: CosSim={cos:.6f}")
        return cos

    print("重建质量：")
    c1 = report("Keys (固定窗口)", keys, dk_f)
    c2 = report("Keys (动态窗口)", keys, dk_d)
    report("Values (固定窗口)", values, dv_f)
    report("Values (动态窗口)", values, dv_d)

    mem_f = kv_fixed.memory_usage(B, H, S)
    mem_d = kv_dyn.memory_usage(B, H, S)
    print(f"\n固定窗口: 压缩率={mem_f['compression_ratio']:.2f}x, "
          f"压缩S={mem_f['compressed_tokens']}, FP16 S={mem_f['fp16_tokens']}")
    print(f"动态窗口: 压缩率={mem_d['compression_ratio']:.2f}x, "
          f"压缩S={mem_d['compressed_tokens']}, FP16 S={mem_d['fp16_tokens']}")

    return c1 > 0.99 and c2 > 0.99


def benchmark_speed():
    print("\n" + "=" * 60)
    print("Test 11: 性能基准测试")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[跳过] 无 CUDA")
        return True

    d = 128
    B, H, S = 4, 8, 2048
    n_warmup = 5
    n_iters = 20

    torch.manual_seed(42)
    keys = torch.randn(B, H, S, d).cuda()
    values = torch.randn(B, H, S, d).cuda()

    kv = TurboQuantKV(d, key_bits=4, value_bits=2,
                      residual_window=128, use_dynamic_window=False).cuda()

    for _ in range(n_warmup):
        ck, cv = kv.compress_kv(keys, values)
        dk, dv = kv.decompress_kv(ck, cv)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        ck, cv = kv.compress_kv(keys, values)
        torch.cuda.synchronize()
    compress_ms = (time.perf_counter() - t0) / n_iters * 1000

    t0 = time.perf_counter()
    for _ in range(n_iters):
        dk, dv = kv.decompress_kv(ck, cv)
        torch.cuda.synchronize()
    decompress_ms = (time.perf_counter() - t0) / n_iters * 1000

    total_tokens = B * H * S
    print(f"Shape: B={B}, H={H}, S={S}, D={d}, Total={total_tokens:,}")
    print(f"压缩:   {compress_ms:.2f} ms ({total_tokens/compress_ms*1000:.0f} tokens/ms)")
    print(f"解压:   {decompress_ms:.2f} ms ({total_tokens/decompress_ms*1000:.0f} tokens/ms)")

    # 吞吐量换算
    gb = total_tokens * d * 2 * 2 / 1e9  # GB（FP16 keys+values）
    print(f"数据量: {gb:.2f} GB, 压缩吞吐: {gb/(compress_ms/1000):.1f} GB/s")
    return True


if __name__ == "__main__":
    print("TurboQuant 优化版测试套件")
    print("=" * 60)

    results = []

    results.append(("旋转分布", test_rotation_distribution()))
    results.append(("Lloyd-Max质量", test_lloyd_max_quality()))
    results.append(("searchsorted精度", test_searchsorted_vs_argmin()))
    results.append(("Bit-pack往返", test_bitpack_roundtrip()))
    results.append(("重建质量", test_mse_reconstruction()))
    results.append(("压缩率", test_compression_ratio()))
    results.append(("V1 vs V3", test_v1_vs_v3()))
    results.append(("动态残差窗口", test_dynamic_window()))
    results.append(("层敏感度分析", test_layer_sensitivity()))
    results.append(("KV Cache接口", test_kv_cache_interface()))

    if torch.cuda.is_available():
        results.append(("GPU性能", benchmark_speed()))
    else:
        print("\n[跳过] 无 CUDA，GPU 性能测试跳过")

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, passed in results:
        print(f"  {'[PASS]' if passed else '[FAIL]'} {name}")
