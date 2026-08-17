from pathlib import Path

p = Path('.github/workflows/test.yml')
text = p.read_text(encoding='utf-8')
old = '      - name: Run Panel Stage B external alignment\n        run: python -m pytest dev/tests/test_panel_stage_b_linearmodels.py -q --tb=short\n'
new = '''      - name: Run maintained panel external alignment
        run: |
          python -m pytest \\
            dev/tests/test_panel_stage_b_linearmodels.py \\
            dev/tests/test_fama_macbeth_linearmodels_external.py \\
            dev/tests/test_panel_pr126_fmb_exact_period.py \\
            -q --tb=short
'''
if old not in text:
    raise RuntimeError('expected panel-stage-b-linearmodels command not found')
p.write_text(text.replace(old, new, 1), encoding='utf-8')
