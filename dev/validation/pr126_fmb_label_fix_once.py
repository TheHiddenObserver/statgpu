from pathlib import Path

path = Path("statgpu/panel/_fama_macbeth.py")
text = path.read_text(encoding="utf-8")
old = '''    for code in eligible_codes:\n        rank = rank_by_code[code]\n        if rank < k:\n            raise ValueError(\n                "FamaMacBeth requires full column rank in every retained period; "\n                f"retained time period {time_labels[code]!r} is rank deficient "\n                f"(rank={rank}, columns={k})"\n            )\n'''
new = '''    for code in eligible_codes:\n        rank = rank_by_code[code]\n        if rank < k:\n            period_label = time_labels[code]\n            if isinstance(period_label, np.generic):\n                period_label = period_label.item()\n            raise ValueError(\n                "FamaMacBeth requires full column rank in every retained period; "\n                f"retained time period {period_label!r} is rank deficient "\n                f"(rank={rank}, columns={k})"\n            )\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("FamaMacBeth rank-error label anchor not found")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
