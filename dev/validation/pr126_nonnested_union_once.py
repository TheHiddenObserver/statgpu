from pathlib import Path

p = Path("dev/validation/pr126_nonnested_union_once.py")
text = p.read_text(encoding="utf-8")
text = text.replace(
    '''                correction1 = _low_order_covariance(grouped1_high, grouped1_low)\n                correction2 = _low_order_covariance(grouped2_high, grouped2_low)\n                correction12 = _low_order_covariance(\n                    grouped12_high, grouped12_low\n                )\n                correction = _stable_inclusion_exclusion(\n                    correction1, correction2, correction12, xp\n                )\n''',
    '''                low_correction1 = _low_order_covariance(\n                    grouped1_high, grouped1_low\n                )\n                low_correction2 = _low_order_covariance(\n                    grouped2_high, grouped2_low\n                )\n                low_correction12 = _low_order_covariance(\n                    grouped12_high, grouped12_low\n                )\n                low_correction = _stable_inclusion_exclusion(\n                    low_correction1, low_correction2, low_correction12, xp\n                )\n''',
)
text = text.replace(
    '''                cov_work = _symmetrize(cov_work + correction)\n''',
    '''                cov_work = _symmetrize(cov_work + low_correction)\n''',
)
p.write_text(text, encoding="utf-8")
