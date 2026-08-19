from pathlib import Path

p = Path("statgpu/panel/_fixed_effects.py")
text = p.read_text(encoding="utf-8")

old = '''            ent_effects_dev, time_effects_dev = _recover_two_way_effects(\n                resid_centered,\n                entity_arr,\n                time_arr,\n                xp,\n                max_iter=self.demean_max_iter,\n                tol=self.demean_tol,\n            )\n            ent_effects = np.asarray(_to_numpy(ent_effects_dev)).ravel()\n'''
new = '''            ent_effects_dev, time_effects_dev = _recover_two_way_effects(\n                resid_orig,\n                entity_arr,\n                time_arr,\n                xp,\n                max_iter=self.demean_max_iter,\n                tol=self.demean_tol,\n            )\n            # Normalize only the compact recovered effect vector. Subtracting\n            # the grand mean observation-by-observation can erase a recoverable\n            # low-order group contribution beside very large residual levels.\n            ent_effects_dev = ent_effects_dev - float(grand_mean)\n            ent_effects = np.asarray(_to_numpy(ent_effects_dev)).ravel()\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("two-way effect recovery normalization anchor not found")
    text = text.replace(old, new, 1)

old = '''                    _compact_group_means(\n                        resid_centered, entity_projection, xp\n                    )\n'''
new = '''                    _compact_group_means(\n                        resid_orig, entity_projection, xp\n                    ) - float(grand_mean)\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("entity effect-map normalization anchor not found")
    text = text.replace(old, new, 1)

old = '''                    _compact_group_means(\n                        resid_centered, time_projection, xp\n                    )\n'''
new = '''                    _compact_group_means(\n                        resid_orig, time_projection, xp\n                    ) - float(grand_mean)\n'''
if new not in text:
    if old not in text:
        raise RuntimeError("time effect-map normalization anchor not found")
    text = text.replace(old, new, 1)

# resid_centered is no longer used by effect-map recovery. Keeping an
# observation-level centering temporary would invite the same precision bug.
text = text.replace("        resid_centered = resid_orig - grand_mean\n", "", 1)
p.write_text(text, encoding="utf-8")

# Tighten the changelog description to record the actual normalization rule.
p = Path("CHANGELOG.md")
text = p.read_text(encoding="utf-8")
old = "- **Panel fixed-effect prediction extreme-scale correctness**: one-way entity/time effect maps now reuse the shared cancellation-safe group-mean reducer instead of raw scatter sums, and two-way additive-effect recovery certifies fast-path convergence with the stable reducer when alternating projections create new dynamic range. This keeps known-label `PanelOLS.predict()` level effects consistent with the already hardened within transformation across NumPy, CuPy, and Torch."
new = "- **Panel fixed-effect prediction extreme-scale correctness**: one-way entity/time effect maps now recover stable group means from uncentered level residuals and apply the grand-mean normalization only on the compact effect vector, avoiding observation-level centering that can erase a recoverable low-order group contribution beside huge residuals. Two-way additive-effect recovery follows the same normalization rule and certifies fast-path convergence with the stable reducer when alternating projections create new dynamic range. This keeps known-label `PanelOLS.predict()` level effects consistent with the already hardened within transformation across NumPy, CuPy, and Torch."
if new not in text:
    if old not in text:
        raise RuntimeError("CHANGELOG FE recovery anchor not found")
    text = text.replace(old, new, 1)
p.write_text(text, encoding="utf-8")
