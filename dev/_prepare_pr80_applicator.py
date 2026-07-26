"""Temporary correction for the one-shot PR #80 applicator."""

from pathlib import Path

path = Path(__file__).with_name("_apply_pr80_review_fixes.py")
text = path.read_text(encoding="utf-8")
old = '''text = text.replace("stats.norm.sf", "norm.sf")
text = text.replace("stats.chi2.sf", "chi2.sf")
if "stats." in text:
    remaining = [
        f"{line_number}: {line.strip()}"
        for line_number, line in enumerate(text.splitlines(), start=1)
        if "stats." in line
    ]
    raise RuntimeError(
        "unconverted scipy.stats use remains in _cox.py:\\n"
        + "\\n".join(remaining)
    )
'''
new = '''text = text.replace("stats.norm.sf", "norm.sf")
text = text.replace("stats.norm.cdf", "norm.cdf")
text = text.replace("stats.norm.ppf", "norm.ppf")
text = text.replace("stats.chi2.sf", "chi2.sf")
remaining = [
    f"{line_number}: {line.strip()}"
    for line_number, line in enumerate(text.splitlines(), start=1)
    if "stats.norm" in line or "stats.chi2" in line
]
if remaining:
    raise RuntimeError(
        "unconverted scipy distribution use remains in _cox.py:\\n"
        + "\\n".join(remaining)
    )
'''
if text.count(old) != 1:
    raise RuntimeError("expected one scipy distribution audit block")
text = text.replace(old, new, 1)

score_write_old = '''    flags=re.DOTALL,
)
path.write_text(text, encoding="utf-8")


# 3. Unified inference distributions and position-safe formula intercept removal.
'''
score_write_new = '''    flags=re.DOTALL,
)
text = text.rstrip() + "\\n"
path.write_text(text, encoding="utf-8")


# 3. Unified inference distributions and position-safe formula intercept removal.
'''
if text.count(score_write_old) != 1:
    raise RuntimeError("expected one penalized Cox score write block")
text = text.replace(score_write_old, score_write_new, 1)

cleanup_old = '''# Restore the maintained workflow, then remove every one-shot helper.
subprocess.run(
    ["git", "checkout", "origin/master", "--", ".github/workflows/test.yml"],
    cwd=ROOT,
    check=True,
)
(ROOT / "dev/_apply_pr80_review_fixes.py").unlink()
(ROOT / ".github/workflows/pr80-review-fix.yml").unlink()
'''
cleanup_new = '''# The GitHub App restores workflow files after the code-only bot commit.
(ROOT / "dev/_apply_pr80_review_fixes.py").unlink()
'''
if text.count(cleanup_old) != 1:
    raise RuntimeError("expected one applicator cleanup block")
text = text.replace(cleanup_old, cleanup_new, 1)
path.write_text(text, encoding="utf-8")
