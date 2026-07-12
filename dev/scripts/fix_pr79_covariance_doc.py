from pathlib import Path

path = Path("docs/en/models/covariance.md")
text = path.read_text(encoding="utf-8")
lines = text.splitlines()
changed = 0
for index, line in enumerate(lines):
    if line.startswith("- **GraphicalLasso**: Block coordinate descent iterates over features"):
        lines[index] = (
            "- **GraphicalLasso**: Block coordinate descent iterates over features, "
            "solving an L1-regularized regression per column via cyclical coordinate "
            "descent with soft-thresholding (up to 1000 inner iterations). Outer "
            "convergence is checked by the maximum absolute covariance update."
        )
        changed += 1
if changed != 1:
    raise RuntimeError(f"expected exactly one GraphicalLasso estimating-equation line, got {changed}")
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
