from pathlib import Path

entries = {
    "CHANGELOG.md": "- Removed universal ElasticNet backend thresholds, coefficient tolerances, and fixed speedup claims that were not established for the current exact-head environment; the model guide now requires workload-specific benchmarking and dtype/solver-specific validation.\n",
    "docs/en/changelog.md": "- Removed universal ElasticNet backend thresholds, coefficient tolerances, and fixed speedup claims that were not established for the current exact-head environment; the model guide now requires workload-specific benchmarking and dtype/solver-specific validation.\n",
    "docs/cn/changelog.md": "- 删除当前 exact-head 环境未能支撑的 ElasticNet 通用后端阈值、统一系数容差与固定加速比；模型文档现要求针对具体工作负载进行 benchmark，并按 dtype/求解路径验证数值一致性。\n",
}

for path, entry in entries.items():
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    marker = "# Changelog\n"
    if text.count(marker) != 1:
        raise RuntimeError(f"{path}: expected one changelog header")
    if entry.strip() not in text:
        text = text.replace(marker, marker + "\n" + entry, 1)
    p.write_text(text, encoding="utf-8")

for path in ("docs/en/models/elastic-net.md", "docs/cn/models/elastic-net.md"):
    text = Path(path).read_text(encoding="utf-8")
    forbidden = (
        "3x - 4.4x",
        "< 3e-8",
        "3x - 4.4x",
        "固定加速比",
    )
    found = [token for token in forbidden if token in text]
    if found:
        raise RuntimeError(f"{path}: stale universal performance claim(s): {found}")
