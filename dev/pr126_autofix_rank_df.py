from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one replacement, found {count}")
    p.write_text(text.replace(old, new), encoding="utf-8")


replace_once(
    "statgpu/panel/_random_effects.py",
    """            if slope_indices.size == 0:\n                resid_within = y_within\n""",
    """            if slope_indices.size == 0:\n                rank_within = 0\n                resid_within = y_within\n""",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    """        df_within = n - k - (n_entities - 1)\n        if df_within <= 0:\n            raise ValueError(\n                f\"Not enough observations for within df: n={n}, k={k}, \"\n                f\"n_entities={n_entities}, df_within={df_within}\"\n            )\n        sigma2_e = rss_within / df_within\n\n        df_between = n_entities - k\n        if df_between <= 0:\n            warnings.warn(\n                f\"Between estimator under-identified: n_entities={n_entities} <= k={k}. \"\n                \"Variance component sigma2_a may be unreliable.\",\n                UserWarning,\n                stacklevel=2,\n            )\n            df_between = max(df_between, 1)\n""",
    """        # Preserve the historical Swamy-Arora parameterization at full\n        # column rank, but count only identified auxiliary-regression directions\n        # in the rank-deficient extension. With an explicit level constant, the\n        # within regression drops that annihilated column and the entity nuisance\n        # rank is N; without a constant it is N-1. These formulas reduce exactly\n        # to n-k-(N-1) and N-k on the historical full-rank paths.\n        within_effect_df = n_entities if has_constant else n_entities - 1\n        df_within = n - int(rank_within) - int(within_effect_df)\n        if df_within <= 0:\n            raise ValueError(\n                \"Not enough observations for within df: \"\n                f\"n={n}, rank_within={rank_within}, \"\n                f\"effect_df={within_effect_df}, df_within={df_within}\"\n            )\n        sigma2_e = rss_within / df_within\n\n        df_between = n_entities - int(rank_between)\n        if df_between <= 0:\n            warnings.warn(\n                \"Between estimator under-identified: \"\n                f\"n_entities={n_entities} <= rank_between={rank_between}. \"\n                \"Variance component sigma2_a may be unreliable.\",\n                UserWarning,\n                stacklevel=2,\n            )\n            df_between = max(df_between, 1)\n""",
)
replace_once(
    "statgpu/panel/_random_effects.py",
    """        resid_gls = y_star - X_star @ beta_gls\n        df_resid = n - k\n        self.df_resid = df_resid\n""",
    """        resid_gls = y_star - X_star @ beta_gls\n        # Full-rank behavior remains n-k. The rank-deficient extension uses\n        # identified quasi-demeaned rank so redundant columns cannot change\n        # scale, HC1 correction, or Student-t degrees of freedom.\n        df_resid = n - int(rank_star)\n        if df_resid <= 0:\n            raise ValueError(\n                \"positive residual degrees of freedom required; \"\n                f\"n={n}, rank={rank_star}\"\n            )\n        self.df_resid = df_resid\n""",
)
replace_once(
    "statgpu/panel/_between.py",
    """        if n <= k:\n            raise ValueError(\n                f\"positive residual degrees of freedom required; groups={n}, parameters={k}\"\n            )\n        df_resid = n - k\n""",
    """        if n <= rank_mean:\n            raise ValueError(\n                \"positive residual degrees of freedom required; \"\n                f\"groups={n}, rank={rank_mean}\"\n            )\n        df_resid = n - int(rank_mean)\n""",
)
replace_once(
    "statgpu/panel/_first_diff.py",
    """        if n <= k:\n            raise ValueError(\n                f\"positive residual degrees of freedom required; n={n}, k={k}\"\n            )\n        df_resid = n - k\n""",
    """        if n <= rank_diff:\n            raise ValueError(\n                \"positive residual degrees of freedom required; \"\n                f\"n={n}, rank={rank_diff}\"\n            )\n        df_resid = n - int(rank_diff)\n""",
)
replace_once(
    "statgpu/panel/_fixed_effects.py",
    "        legacy_df_resid = n - k - n_effects\n",
    "        legacy_df_resid = n - int(fit_rank) - n_effects\n",
)
replace_once(
    "statgpu/panel/_base.py",
    """        diag = xp.diag(cov_params)\n        diag_np = np.asarray(_to_numpy(diag), dtype=np.float64).ravel()\n        negative_tol = (\n            4096.0\n            * np.finfo(np.float64).eps\n            * np.maximum(1.0, np.abs(diag_np))\n        )\n""",
    """        diag = xp.diag(cov_params)\n        cov_np = self._panel_cov_params_raw\n        diag_np = np.diag(cov_np).astype(np.float64, copy=False)\n        row_scale = np.max(np.abs(cov_np), axis=1)\n        col_scale = np.max(np.abs(cov_np), axis=0)\n        local_scale = np.maximum(row_scale, col_scale)\n        negative_tol = 4096.0 * np.finfo(np.float64).eps * local_scale\n""",
)

rank_test = Path("dev/tests/test_panel_stage_c_rank_deficient_matrix.py")
text = rank_test.read_text(encoding="utf-8")
addition = r'''

@pytest.mark.parametrize("cov_type", ["nonrobust", "robust"])
def test_rank_deficient_df_and_identified_inference_are_column_space_invariant(cov_type):
    """Redundant columns cannot change identified fit/inference in rank extensions."""
    rng = np.random.default_rng(12913)
    n_entities, n_times = 18, 5
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    keep = np.ones(entity.size, dtype=bool)
    keep[[1, 8, 17, 29, 44, 63, 78]] = False
    entity = entity[keep]
    time = time[keep]
    x = rng.normal(size=entity.size)
    alpha = np.repeat(rng.normal(scale=0.35, size=n_entities), n_times)[keep]
    y = 0.45 + 0.75 * x + alpha + rng.normal(scale=0.2, size=entity.size)

    X1 = x[:, None]
    X2 = np.column_stack([x, 2.0 * x])
    pairs = [
        (PanelOLS(cov_type=cov_type).fit(X1, y),
         PanelOLS(cov_type=cov_type).fit(X2, y), X1, X2),
        (BetweenOLS(cov_type=cov_type).fit(X1, y, entity_ids=entity),
         BetweenOLS(cov_type=cov_type).fit(X2, y, entity_ids=entity),
         np.column_stack([np.ones(len(y)), X1]),
         np.column_stack([np.ones(len(y)), X2])),
        (FirstDifferenceOLS(cov_type=cov_type).fit(X1, y, entity_ids=entity, time_ids=time),
         FirstDifferenceOLS(cov_type=cov_type).fit(X2, y, entity_ids=entity, time_ids=time),
         X1, X2),
    ]

    X1_re = np.column_stack([np.ones(len(y)), X1])
    X2_re = np.column_stack([np.ones(len(y)), X2])
    re_base = RandomEffects(cov_type=cov_type).fit(X1_re, y, entity_ids=entity)
    re_redundant = RandomEffects(cov_type=cov_type).fit(X2_re, y, entity_ids=entity)
    pairs.append((re_base, re_redundant, X1_re, X2_re))

    for base, redundant, design_base, design_redundant in pairs:
        assert base.df_resid == redundant.df_resid
        assert_allclose(
            design_redundant @ np.asarray(redundant.coef_),
            design_base @ np.asarray(base.coef_),
            rtol=2e-10, atol=2e-11,
        )
        cov_base = np.asarray(base._panel_cov_params_raw)
        cov_redundant = np.asarray(redundant._panel_cov_params_raw)
        assert_allclose(
            design_redundant @ cov_redundant @ design_redundant.T,
            design_base @ cov_base @ design_base.T,
            rtol=2e-8, atol=2e-10,
        )

    assert_allclose(re_redundant.variance_components_["sigma2_e"],
                    re_base.variance_components_["sigma2_e"], rtol=2e-11, atol=2e-13)
    assert_allclose(re_redundant.variance_components_["sigma2_a"],
                    re_base.variance_components_["sigma2_a"], rtol=2e-11, atol=2e-13)
    assert_allclose(re_redundant.theta_, re_base.theta_, rtol=2e-11, atol=2e-13)
'''
if "test_rank_deficient_df_and_identified_inference_are_column_space_invariant" in text:
    raise SystemExit("rank-deficient invariance test already exists")
rank_test.write_text(text + addition, encoding="utf-8")

guard_test = Path("dev/tests/test_panel_stage_c_inference_guard.py")
text = guard_test.read_text(encoding="utf-8")
addition = r'''


def _store_with_mock_covariance(monkeypatch, covariance):
    def _fake_covariance(*args, **kwargs):
        return np.asarray(covariance, dtype=np.float64)

    monkeypatch.setattr(_covariance, "ols_covariance", _fake_covariance)
    model = _DummyPanelModel()
    backend = model._get_backend(backend="auto")
    model._panel_store_ols_inference(
        np.eye(2), np.zeros(2), np.ones(2), scale=1.0, df_resid=2,
        backend=backend, cov_type="hc0", allowed=("hc0",), diag_floor=1.0e-30,
    )
    return model


def test_negative_variance_guard_is_scale_equivariant(monkeypatch):
    """Changing outcome units cannot change accept-vs-fail covariance validity."""
    material = np.array([[4.0e-14, 0.0], [0.0, -1.0e-14]])
    for multiplier in (1.0, 1.0e12):
        with pytest.raises(ValueError, match="materially negative diagonal variance"):
            _store_with_mock_covariance(monkeypatch, material * multiplier)

    roundoff = np.array([[4.0e-14, 2.0e-14], [2.0e-14, -1.0e-28]])
    for multiplier in (1.0, 1.0e12):
        model = _store_with_mock_covariance(monkeypatch, roundoff * multiplier)
        assert np.all(np.isfinite(model.bse_))
        assert model.bse_[1] >= 0.0
'''
if "test_negative_variance_guard_is_scale_equivariant" in text:
    raise SystemExit("scale-equivariant guard test already exists")
guard_test.write_text(text + addition, encoding="utf-8")
