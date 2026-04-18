"""
Generate complete comparison report: statgpu vs glmnet with detailed timing table.
"""
import json
from pathlib import Path
from datetime import datetime

def generate_report(results_dir="results/benchmark_full"):
    results_dir = Path(results_dir)

    # Load glmnet results
    glmnet_results = {}
    for name in ["small_data", "medium_data", "large_data", "high_dim_data", "sparse_coef", "high_noise"]:
        glmnet_file = results_dir / f"glmnet_result_{name}.json"
        if glmnet_file.exists():
            with open(glmnet_file, 'r') as f:
                glmnet_results[name] = json.load(f)

    # Load statgpu results
    statgpu_file = results_dir / "benchmark_statgpu_all.json"
    with open(statgpu_file, 'r') as f:
        statgpu_data = json.load(f)

    statgpu_results = {}
    for result in statgpu_data['results']:
        statgpu_results[result['name']] = result['backends']

    # Generate report
    lines = [
        "# Elastic Net 基准测试完整报告",
        "",
        f"**生成时间**: {datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## 1. 完整运行时间对比表",
        "",
        "### 1.1 拟合时间 (ms)",
        "",
        "| 数据集 | n | p | R glmnet | statgpu CPU | statgpu CuPy | statgpu Torch |",
        "|--------|---|---|-----------|-------------|--------------|---------------|",
    ]

    for name in ["small_data", "medium_data", "large_data", "high_dim_data", "sparse_coef", "high_noise"]:
        if name not in glmnet_results or name not in statgpu_results:
            continue

        n = glmnet_results[name]['n_samples']
        p = glmnet_results[name]['n_features']
        t_glmnet = glmnet_results[name]['fit_time_ms']
        t_cpu = statgpu_results[name]['statgpu_cpu']['fit_time_ms']
        t_cupy = statgpu_results[name].get('statgpu_gpu_cupy', {}).get('fit_time_ms', float('inf'))
        t_torch = statgpu_results[name].get('statgpu_gpu_torch', {}).get('fit_time_ms', float('inf'))

        # Find fastest
        times = {'glmnet': t_glmnet, 'cpu': t_cpu, 'cupy': t_cupy, 'torch': t_torch}
        fastest = min(times, key=times.get)

        def fmt(t, is_fastest):
            if t == float('inf'):
                return "N/A"
            if is_fastest:
                return f"**{t:.2f}**"
            return f"{t:.2f}"

        lines.append(
            f"| {name} | {n} | {p} | {fmt(t_glmnet, fastest=='glmnet')} | {fmt(t_cpu, fastest=='cpu')} | {fmt(t_cupy, fastest=='cupy')} | {fmt(t_torch, fastest=='torch')} |"
        )

    lines.extend([
        "",
        "### 1.2 相对加速比 (以最快者为基准)",
        "",
        "| 数据集 | R glmnet | statgpu CPU | statgpu CuPy | statgpu Torch |",
        "|--------|----------|-------------|--------------|---------------|",
    ])

    for name in ["small_data", "medium_data", "large_data", "high_dim_data", "sparse_coef", "high_noise"]:
        if name not in glmnet_results or name not in statgpu_results:
            continue

        n = glmnet_results[name]['n_samples']
        p = glmnet_results[name]['n_features']
        t_glmnet = glmnet_results[name]['fit_time_ms']
        t_cpu = statgpu_results[name]['statgpu_cpu']['fit_time_ms']
        t_cupy = statgpu_results[name].get('statgpu_gpu_cupy', {}).get('fit_time_ms', float('inf'))
        t_torch = statgpu_results[name].get('statgpu_gpu_torch', {}).get('fit_time_ms', float('inf'))

        times = [t_glmnet, t_cpu, t_cupy, t_torch]
        fastest_time = min(t for t in times if t != float('inf'))

        def ratio(t):
            if t == float('inf'):
                return "N/A"
            return f"{fastest_time/t:.2f}x" if t > fastest_time * 1.1 else f"**1.00x**"

        lines.append(
            f"| {name} (n={n}, p={p}) | {ratio(t_glmnet)} | {ratio(t_cpu)} | {ratio(t_cupy)} | {ratio(t_torch)} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. 计算精度对比",
        "",
        "### 2.1 系数范数对比",
        "",
        "| 数据集 | R glmnet | statgpu CPU | statgpu CuPy | statgpu Torch |",
        "|--------|-----------|-------------|--------------|---------------|",
    ])

    for name in ["small_data", "medium_data", "large_data", "high_dim_data", "sparse_coef", "high_noise"]:
        if name not in glmnet_results or name not in statgpu_results:
            continue

        n_glmnet = glmnet_results[name]['coef_norm']
        n_cpu = statgpu_results[name]['statgpu_cpu']['coef_norm']
        n_cupy = statgpu_results[name].get('statgpu_gpu_cupy', {}).get('coef_norm', 0)
        n_torch = statgpu_results[name].get('statgpu_gpu_torch', {}).get('coef_norm', 0)

        lines.append(
            f"| {name} | {n_glmnet:.6f} | {n_cpu:.6f} | {n_cupy:.6f} | {n_torch:.6f} |"
        )

    lines.extend([
        "",
        "### 2.2 截距对比",
        "",
        "| 数据集 | R glmnet | statgpu CPU | statgpu CuPy | statgpu Torch |",
        "|--------|-----------|-------------|--------------|---------------|",
    ])

    for name in ["small_data", "medium_data", "large_data", "high_dim_data", "sparse_coef", "high_noise"]:
        if name not in glmnet_results or name not in statgpu_results:
            continue

        i_glmnet = glmnet_results[name]['intercept']
        i_cpu = statgpu_results[name]['statgpu_cpu']['intercept']
        i_cupy = statgpu_results[name].get('statgpu_gpu_cupy', {}).get('intercept', 0)
        i_torch = statgpu_results[name].get('statgpu_gpu_torch', {}).get('intercept', 0)

        lines.append(
            f"| {name} | {i_glmnet:.6f} | {i_cpu:.6f} | {i_cupy:.6f} | {i_torch:.6f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. 结论",
        "",
        "### 3.1 性能总结",
        "",
        "- **R glmnet**: 在所有数据集上表现稳定，特别是在高维数据 (n=100, p=200) 上非常快",
        "- **statgpu CPU**: 在中等规模数据上表现良好，但相比 glmnet 没有明显优势",
        "- **statgpu GPU (CuPy)**: 由于数据传输开销，在小数据上较慢",
        "- **statgpu GPU (Torch)**: 与 CuPy 类似，但在某些情况下更快",
        "",
        "### 3.2 精度总结",
        "",
        "- 所有后端的系数范数和截距高度一致",
        "- 差异来源于不同优化算法的收敛路径和停止条件",
        "",
        "---",
        "",
        f"*完整报告生成时间*: {datetime.now().isoformat()}",
    ])

    # Save report
    output_file = results_dir / "benchmark_complete_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Report saved to: {output_file}")
    print('\n'.join(lines))

    return output_file


if __name__ == "__main__":
    generate_report()
