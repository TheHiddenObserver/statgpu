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
path.write_text(text.replace(old, new, 1), encoding="utf-8")
