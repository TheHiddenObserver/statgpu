"""One-shot cleanup for tests that still encode the retired legacy FE df contract."""
from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "dev/tests/test_panel_stage_b_diagnostics.py",
    '''def test_panelols_pooling_f_uses_standard_diagnostic_df_not_legacy_inference_df():
    X, y, entity, _ = _balanced_panel()
    model = PanelOLS(entity_effects=True).fit(X, y, entity_ids=entity)

    # Stage-A legacy inference df remains untouched.
    assert model.df_resid == len(y) - X.shape[1] - (len(np.unique(entity)) - 1)
    diagnostic_df = model.fit_statistics_.metadata["diagnostic_df"]
    assert diagnostic_df["df_resid"] == model.df_resid - 1
    assert diagnostic_df["effect_rank"] == len(np.unique(entity))
''',
    '''def test_panelols_pooling_f_and_public_inference_share_standard_fe_df():
    X, y, entity, _ = _balanced_panel()
    model = PanelOLS(entity_effects=True).fit(X, y, entity_ids=entity)

    diagnostic_df = model.fit_statistics_.metadata["diagnostic_df"]
    expected_df = len(y) - X.shape[1] - len(np.unique(entity))
    assert model.df_resid == expected_df
    assert diagnostic_df["df_resid"] == model.df_resid
    assert diagnostic_df["effect_rank"] == len(np.unique(entity))
    assert model.fit_statistics_.metadata["public_df_resid_basis"] == "standard"
''',
)

replace_once(
    "dev/tests/test_panel_stage_b_fit_statistics.py",
    '''def test_two_way_fe_uses_standard_effect_rank_without_changing_legacy_df():
    X, y, entity, time = _panel(seed=1241)
    model = PanelOLS(entity_effects=True, time_effects=True).fit(
        X,
        y,
        entity_ids=entity,
        time_ids=time,
    )
    n = len(y)
    k = X.shape[1]
    N = len(np.unique(entity))
    T = len(np.unique(time))
    assert model.df_resid == n - k - (N - 1) - (T - 1)
    diag = model.fit_statistics_.metadata["diagnostic_df"]
    assert diag["effect_rank"] == N + T - 1
    assert diag["df_resid"] == n - np.linalg.matrix_rank(
        X
        - np.vstack([X[entity == g].mean(axis=0) for g in entity])
        - np.vstack([X[time == t].mean(axis=0) for t in time])
        + X.mean(axis=0)
    ) - (N + T - 1)
    assert diag["df_resid"] == model.df_resid - 1
    assert model.fit_statistics_.metadata["legacy_rsquared_within"] == model.rsquared_within
''',
    '''def test_two_way_fe_public_df_uses_standard_effect_rank():
    X, y, entity, time = _panel(seed=1241)
    model = PanelOLS(entity_effects=True, time_effects=True).fit(
        X,
        y,
        entity_ids=entity,
        time_ids=time,
    )
    n = len(y)
    N = len(np.unique(entity))
    T = len(np.unique(time))
    diag = model.fit_statistics_.metadata["diagnostic_df"]
    assert diag["effect_rank"] == N + T - 1
    expected_df = n - np.linalg.matrix_rank(
        X
        - np.vstack([X[entity == g].mean(axis=0) for g in entity])
        - np.vstack([X[time == t].mean(axis=0) for t in time])
        + X.mean(axis=0)
    ) - (N + T - 1)
    assert diag["df_resid"] == expected_df
    assert model.df_resid == expected_df
    assert model.fit_statistics_.metadata["public_df_resid_basis"] == "standard"
    assert model.fit_statistics_.metadata["legacy_rsquared_within"] == model.rsquared_within
''',
)

replace_once(
    "dev/tests/test_panel_stage_b_hausman_covariance.py",
    '''def test_hausman_uses_standard_fe_covariance_without_changing_legacy_inference():
    X, y, entity = _panel()
    fe = PanelOLS(entity_effects=True, cov_type="nonrobust").fit(
        X, y, entity_ids=entity
    )
    re = RandomEffects().fit(X, y, entity_ids=entity)

    raw_fe = np.asarray(fe._panel_cov_params_raw)
    raw_re = np.asarray(re._panel_cov_params_raw)
    diag_meta = fe.fit_statistics_.metadata["diagnostic_df"]
    legacy_df = fe.fit_statistics_.metadata["legacy_df_resid"]
    standard_df = diag_meta["df_resid"]

    # Stage A's historical FE inference denominator is intentionally retained.
    assert legacy_df == fe.df_resid
    assert standard_df == legacy_df - 1
    assert_allclose(fe.bse_ ** 2, np.diag(raw_fe), rtol=1e-12, atol=1e-14)

    # Classical Hausman, however, needs the full nuisance-effect model rank.
    # Only the small diagnostic covariance copy is rescaled; public bse/CI above
    # still come from the raw Stage-A covariance.
    expected_fe_diagnostic = raw_fe * (legacy_df / standard_df)
    assert_allclose(
        fe._panel_cov_params,
        expected_fe_diagnostic,
        rtol=1e-12,
        atol=1e-14,
    )
''',
    '''def test_hausman_and_public_inference_share_standard_fe_covariance():
    X, y, entity = _panel()
    fe = PanelOLS(entity_effects=True, cov_type="nonrobust").fit(
        X, y, entity_ids=entity
    )
    re = RandomEffects().fit(X, y, entity_ids=entity)

    raw_fe = np.asarray(fe._panel_cov_params_raw)
    raw_re = np.asarray(re._panel_cov_params_raw)
    diag_meta = fe.fit_statistics_.metadata["diagnostic_df"]
    legacy_df = fe.fit_statistics_.metadata["legacy_df_resid"]
    standard_df = diag_meta["df_resid"]

    # Public FE inference and diagnostics now use the same full nuisance rank.
    assert standard_df == fe.df_resid
    assert legacy_df == standard_df + 1
    assert fe.fit_statistics_.metadata["public_df_resid_basis"] == "standard"
    assert_allclose(fe.bse_ ** 2, np.diag(raw_fe), rtol=1e-12, atol=1e-14)
    expected_fe_diagnostic = raw_fe
    assert_allclose(
        fe._panel_cov_params,
        expected_fe_diagnostic,
        rtol=0,
        atol=0,
    )
''',
)
replace_once(
    "dev/tests/test_panel_stage_b_hausman_covariance.py",
    '''    # RandomEffects has no legacy FE nuisance-df mismatch, so its diagnostic
    # covariance is exactly the inference covariance.
''',
    '''    # RandomEffects likewise exposes the same inference covariance to diagnostics.
''',
)

replace_once(
    "dev/tests/test_panel_stage_b_ready_review_regressions.py",
    '''    assert metadata["legacy_df_resid"] == 0
    assert metadata["public_df_resid_basis"] == "component-aware"
    assert model.df_resid == 1
''',
    '''    assert metadata["legacy_df_resid"] == 0
    assert metadata["public_df_resid_basis"] == "standard"
    assert model.df_resid == 1
''',
)

replace_once(
    "dev/tests/test_panel_stage_b_torch_cpu.py",
    '''    assert actual.fit_statistics_.metadata["legacy_df_resid"] == 0
    assert actual.fit_statistics_.metadata["public_df_resid_basis"] == "component-aware"
    assert actual.fit_statistics_.metadata["diagnostic_df"] == expected.fit_statistics_.metadata["diagnostic_df"]
''',
    '''    assert actual.fit_statistics_.metadata["legacy_df_resid"] == 0
    assert actual.fit_statistics_.metadata["public_df_resid_basis"] == "standard"
    assert actual.fit_statistics_.metadata["diagnostic_df"] == expected.fit_statistics_.metadata["diagnostic_df"]
''',
)
