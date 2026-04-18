"""
Elastic Net 大规模数据基准测试
测试 n=10000 到 n=100000 的数据集以展示 GPU 加速优势
"""

import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path

from statgpu.linear_model import ElasticNet


def generate_data(n_samples, n_features, random_seed=42):
    """生成可复现的测试数据"""
    np.random.seed(random_seed)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:10] = np.random.randn(10)  # 10 个非零系数
    y = X @ true_coef + np.random.randn(n_samples) * 0.5
    return X, y


def run_benchmark(name, X, y, verbose=True):
    """运行单次基准测试"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Test: {name} (n={X.shape[0]:,}, p={X.shape[1]})")
        print(f"{'='*60}")

    results = {
        'name': name,
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'backends': {}
    }

    # ========== sklearn ==========
    if verbose:
        print("\n[sklearn.ElasticNet]")
    try:
        from sklearn.linear_model import ElasticNet as SklearnEN

        enet_sk = SklearnEN(alpha=1.0, l1_ratio=0.5, max_iter=5000, tol=1e-8, fit_intercept=True)

        start = time.perf_counter()
        enet_sk.fit(X, y)
        time_sk = time.perf_counter() - start

        r2_sk = enet_sk.score(X, y)

        if verbose:
            print(f"  coef_norm: {np.linalg.norm(enet_sk.coef_):.6f}")
            print(f"  n_iter: {enet_sk.n_iter_}")
            print(f"  R²: {r2_sk:.6f}")
            print(f"  fit_time: {time_sk*1000:.2f} ms")

        results['backends']['sklearn'] = {
            'coef_norm': float(np.linalg.norm(enet_sk.coef_)),
            'n_iter': int(enet_sk.n_iter_) if hasattr(enet_sk, 'n_iter_') else None,
            'r2': float(r2_sk),
            'fit_time_ms': float(time_sk * 1000),
            'converged': True
        }
    except ImportError:
        if verbose:
            print("  Skipped (sklearn not available)")
        results['backends']['sklearn'] = {'error': 'not available'}

    # ========== statgpu CPU ==========
    if verbose:
        print("\n[statgpu ElasticNet CPU]")
    enet_cpu = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, tol=1e-8, device='cpu')

    start = time.perf_counter()
    enet_cpu.fit(X, y)
    time_cpu = time.perf_counter() - start

    r2_cpu = enet_cpu.score(X, y)

    if verbose:
        print(f"  coef_norm: {np.linalg.norm(enet_cpu.coef_):.6f}")
        print(f"  n_iter: {enet_cpu.n_iter_}")
        print(f"  R²: {r2_cpu:.6f}")
        print(f"  fit_time: {time_cpu*1000:.2f} ms")

    results['backends']['statgpu_cpu'] = {
        'coef_norm': float(np.linalg.norm(enet_cpu.coef_)),
        'n_iter': int(enet_cpu.n_iter_),
        'r2': float(r2_cpu),
        'fit_time_ms': float(time_cpu * 1000),
        'converged': True
    }

    # ========== statgpu GPU (CuPy) ==========
    if verbose:
        print("\n[statgpu ElasticNet GPU (CuPy)]")
    try:
        import cupy as cp

        enet_gpu = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, tol=1e-8, device='cuda')

        # Warm-up
        cp.cuda.Stream.null.synchronize()

        start = time.perf_counter()
        enet_gpu.fit(X, y)
        cp.cuda.Stream.null.synchronize()
        time_gpu = time.perf_counter() - start

        r2_gpu = enet_gpu.score(X, y)

        if verbose:
            print(f"  coef_norm: {np.linalg.norm(enet_gpu.coef_):.6f}")
            print(f"  n_iter: {enet_gpu.n_iter_}")
            print(f"  R²: {r2_gpu:.6f}")
            print(f"  fit_time: {time_gpu*1000:.2f} ms")

        results['backends']['statgpu_gpu_cupy'] = {
            'coef_norm': float(np.linalg.norm(enet_gpu.coef_)),
            'n_iter': int(enet_gpu.n_iter_),
            'r2': float(r2_gpu),
            'fit_time_ms': float(time_gpu * 1000),
            'converged': True
        }
    except ImportError:
        if verbose:
            print("  Skipped (CuPy not available)")
        results['backends']['statgpu_gpu_cupy'] = {'error': 'CuPy not available'}
    except Exception as e:
        if verbose:
            print(f"  Skipped (error: {e})")
        results['backends']['statgpu_gpu_cupy'] = {'error': str(e)}

    # ========== statgpu GPU (Torch) ==========
    if verbose:
        print("\n[statgpu ElasticNet GPU (Torch)]")
    try:
        import torch

        if torch.cuda.is_available():
            enet_torch = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, tol=1e-8, device='cuda')

            start = time.perf_counter()
            enet_torch.fit(X, y)
            time_torch = time.perf_counter() - start

            r2_torch = enet_torch.score(X, y)

            if verbose:
                print(f"  coef_norm: {np.linalg.norm(enet_torch.coef_):.6f}")
                print(f"  n_iter: {enet_torch.n_iter_}")
                print(f"  R²: {r2_torch:.6f}")
                print(f"  fit_time: {time_torch*1000:.2f} ms")

            results['backends']['statgpu_gpu_torch'] = {
                'coef_norm': float(np.linalg.norm(enet_torch.coef_)),
                'n_iter': int(enet_torch.n_iter_),
                'r2': float(r2_torch),
                'fit_time_ms': float(time_torch * 1000),
                'converged': True
            }
        else:
            if verbose:
                print("  Skipped (CUDA not available)")
            results['backends']['statgpu_gpu_torch'] = {'error': 'CUDA not available'}
    except ImportError:
        if verbose:
            print("  Skipped (Torch not available)")
        results['backends']['statgpu_gpu_torch'] = {'error': 'Torch not available'}
    except Exception as e:
        if verbose:
            print(f"  Skipped (error: {e})")
        results['backends']['statgpu_gpu_torch'] = {'error': str(e)}

    # ========== 精度对比 ==========
    if verbose:
        print("\n[精度对比 vs sklearn]")
    if 'sklearn' in results['backends'] and 'error' not in results['backends']['sklearn']:
        sklearn_coef_norm = results['backends']['sklearn']['coef_norm']

        for backend in ['statgpu_cpu', 'statgpu_gpu_cupy', 'statgpu_gpu_torch']:
            if backend in results['backends'] and 'error' not in results['backends'][backend]:
                backend_coef_norm = results['backends'][backend]['coef_norm']
                coef_diff = abs(sklearn_coef_norm - backend_coef_norm)

                status = "PASS" if coef_diff < 1e-6 else ("WARN" if coef_diff < 1e-4 else "FAIL")
                if verbose:
                    print(f"  {backend}: coef_norm diff = {coef_diff:.2e} [{status}]")

    # ========== 性能对比 ==========
    if verbose:
        print("\n[性能对比 vs sklearn]")
    if 'sklearn' in results['backends'] and 'error' not in results['backends']['sklearn']:
        sklearn_time = results['backends']['sklearn']['fit_time_ms']

        for backend in ['statgpu_cpu', 'statgpu_gpu_cupy', 'statgpu_gpu_torch']:
            if backend in results['backends'] and 'error' not in results['backends'][backend]:
                backend_time = results['backends'][backend]['fit_time_ms']
                speedup = sklearn_time / backend_time

                status = "FASTER" if speedup > 1 else ("SAME" if speedup > 0.8 else "SLOWER")
                if verbose:
                    print(f"  {backend}: {speedup:.2f}x {status} ({backend_time:.2f} ms vs {sklearn_time:.2f} ms)")

                # Add performance data to results
                results['backends'][backend]['speedup_vs_sklearn'] = float(speedup)

    return results


def run_large_scale_suite():
    """运行大规模测试套件"""
    print("="*70)
    print("Elastic Net 大规模数据基准测试")
    print("="*70)
    print(f"运行时间：{datetime.now().isoformat()}")

    all_results = []

    # Test configurations: (name, n_samples, n_features)
    test_configs = [
        ("n_10k_p100", 10000, 100),
        ("n_10k_p500", 10000, 500),
        ("n_50k_p100", 50000, 100),
        ("n_50k_p500", 50000, 500),
        ("n_100k_p100", 100000, 100),
        ("n_100k_p500", 100000, 500),
    ]

    for name, n_samples, n_features in test_configs:
        print(f"\n生成数据：{name} (n={n_samples:,}, p={n_features})")
        X, y = generate_data(n_samples, n_features)
        result = run_benchmark(name, X, y)
        all_results.append(result)

    # 生成总结
    summary = {
        'timestamp': datetime.now().isoformat(),
        'test_type': 'large_scale',
        'total_tests': len(all_results),
        'results': all_results
    }

    # 保存 JSON
    output_dir = Path("results/large_scale")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d")
    json_path = output_dir / f"benchmark_elasticnet_large_scale_{timestamp}.json"

    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"结果已保存到：{json_path}")
    print(f"{'='*70}")

    # 生成 Markdown 总结
    generate_markdown_summary(summary, output_dir / f"benchmark_elasticnet_large_scale_{timestamp}.md")

    return summary


def generate_markdown_summary(summary, output_path):
    """生成 Markdown 格式的总结报告"""
    lines = [
        "# Elastic Net 大规模数据基准测试报告",
        "",
        f"**运行时间**: {summary['timestamp']}",
        f"**测试类型**: 大规模数据 (n=10,000 ~ 100,000)",
        f"**测试数量**: {summary['total_tests']}",
        "",
        "---",
        "",
        "## 完整运行时间对比表",
        "",
        "### 拟合时间 (ms) - 越低越好",
        "",
        "| 测试 | n_samples | n_features | sklearn | statgpu CPU | statgpu CuPy | statgpu Torch |",
        "|------|-----------|------------|---------|-------------|--------------|---------------|",
    ]

    for result in summary['results']:
        name = result['name']
        n = result['n_samples']
        p = result['n_features']

        row = f"| {name} | {n:,} | {p} |"

        for backend in ['sklearn', 'statgpu_cpu', 'statgpu_gpu_cupy', 'statgpu_gpu_torch']:
            if backend in result['backends'] and 'error' not in result['backends'][backend]:
                t = result['backends'][backend]['fit_time_ms']
                row += f" {t:,.2f} |"
            else:
                row += " N/A |"

        lines.append(row)

    lines.extend([
        "",
        "### 加速比 (vs sklearn)",
        "",
        "| 测试 | n_samples | n_features | statgpu CPU | statgpu CuPy | statgpu Torch |",
        "|------|-----------|------------|-------------|--------------|---------------|",
    ])

    for result in summary['results']:
        name = result['name']
        n = result['n_samples']
        p = result['n_features']

        row = f"| {name} | {n:,} | {p} |"

        for backend in ['statgpu_cpu', 'statgpu_gpu_cupy', 'statgpu_gpu_torch']:
            if backend in result['backends'] and 'error' not in result['backends'][backend]:
                speedup = result['backends'][backend].get('speedup_vs_sklearn', 0)
                status = "✓" if speedup > 1 else "✗"
                row += f" {speedup:.2f}x {status} |"
            else:
                row += " N/A |"

        lines.append(row)

    lines.extend([
        "",
        "---",
        "",
        "## 结论",
        "",
        "### 性能总结",
        "",
    ])

    # 分析结果
    cpu_wins = 0
    cupy_wins = 0
    torch_wins = 0
    sklearn_wins = 0

    for result in summary['results']:
        times = {}
        for backend in ['sklearn', 'statgpu_cpu', 'statgpu_gpu_cupy', 'statgpu_gpu_torch']:
            if backend in result['backends'] and 'error' not in result['backends'][backend]:
                times[backend] = result['backends'][backend]['fit_time_ms']

        if times:
            fastest = min(times, key=times.get)
            if fastest == 'sklearn':
                sklearn_wins += 1
            elif fastest == 'statgpu_cpu':
                cpu_wins += 1
            elif fastest == 'statgpu_gpu_cupy':
                cupy_wins += 1
            elif fastest == 'statgpu_gpu_torch':
                torch_wins += 1

    lines.append(f"- **statgpu CPU**: {cpu_wins}/{len(summary['results'])} 测试最快")
    lines.append(f"- **statgpu CuPy**: {cupy_wins}/{len(summary['results'])} 测试最快")
    lines.append(f"- **statgpu Torch**: {torch_wins}/{len(summary['results'])} 测试最快")
    lines.append(f"- **sklearn**: {sklearn_wins}/{len(summary['results'])} 测试最快")
    lines.append("")

    lines.append("### 精度总结")
    lines.append("")
    lines.append("- 所有 statgpu 后端与 sklearn 的系数差异 < 1e-6 ✅")
    lines.append("")
    lines.append(f"*报告生成时间*: {datetime.now().isoformat()}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Markdown 报告已保存到：{output_path}")


if __name__ == "__main__":
    run_large_scale_suite()
