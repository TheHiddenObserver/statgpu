"""One-shot helper for the fresh PR126 FE-df/rank-guard review fixes."""
from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    p.write_text(text.replace(old, new, 1))


# ---------------------------------------------------------------------------
# 1) PanelOLS: public df/inference must use the same standard FE rank already
#    used by diagnostics and DK. Keep legacy df only as metadata.
# ---------------------------------------------------------------------------
replace_once(
    "statgpu/panel/_fixed_effects.py",
    '''        legacy_df_resid = n - int(fit_rank) - n_effects
        standard_df_resid = int(diagnostic_df["df_resid"])
        if legacy_df_resid > 0:
            self.df_resid = legacy_df_resid
            public_df_basis = "legacy"
        elif standard_df_resid > 0:
            self.df_resid = standard_df_resid
            public_df_basis = "component-aware"
        else:
            raise ValueError(
                "Not enough observations after fixed-effect rank adjustment: "
                f"n={n}, k={k}, legacy_n_effects={n_effects}, "
                f"legacy_df_resid={legacy_df_resid}, "
                f"effect_rank={diagnostic_df['effect_rank']}, "
                f"incidence_components={diagnostic_df['incidence_components']}, "
                f"df_resid={standard_df_resid}."
            )
''',
    '''        legacy_df_resid = n - int(fit_rank) - n_effects
        standard_df_resid = int(diagnostic_df["df_resid"])
        if standard_df_resid <= 0:
            raise ValueError(
                "Not enough observations after fixed-effect rank adjustment: "
                f"n={n}, k={k}, legacy_n_effects={n_effects}, "
                f"legacy_df_resid={legacy_df_resid}, "
                f"effect_rank={diagnostic_df['effect_rank']}, "
                f"incidence_components={diagnostic_df['incidence_components']}, "
                f"df_resid={standard_df_resid}."
            )
        # Public inference must count the full identified nuisance-effect rank.
        # The older N-1/T-1 shortcut omitted one nuisance direction whenever FE
        # were represented by within transformation without an explicit level
        # constant, understating nonrobust scale and the HC1 correction.
        self.df_resid = standard_df_resid
        public_df_basis = "standard"
''',
)

# The diagnostic covariance compatibility property should only rescale an
# actually legacy-scaled stored covariance. New PanelOLS fits are standard.
replace_once(
    "statgpu/panel/_base.py",
    '''        diagnostic_df = metadata.get("diagnostic_df")
        legacy_df = metadata.get("legacy_df_resid")
        cov_type = str(getattr(self, "_cov_type", "nonrobust")).lower()
        if (
            cov_type == "nonrobust"
            and isinstance(diagnostic_df, dict)
            and legacy_df is not None
        ):
''',
    '''        diagnostic_df = metadata.get("diagnostic_df")
        legacy_df = metadata.get("legacy_df_resid")
        public_df_basis = metadata.get("public_df_resid_basis")
        cov_type = str(getattr(self, "_cov_type", "nonrobust")).lower()
        if (
            cov_type == "nonrobust"
            and public_df_basis == "legacy"
            and isinstance(diagnostic_df, dict)
            and legacy_df is not None
        ):
''',
)
replace_once(
    "statgpu/panel/_base.py",
    '''        Existing Stage-A inference is computed and stored before this property is
        consulted, so rescaling here cannot change public bse/t/p/CI values.
        PanelOLS preserves a historical residual-df convention that is one rank
        parameterization away from the standard fixed-effect model df used by
        classical Hausman tests.  When Stage-B fit metadata provides both the
        legacy and standard diagnostic df, convert only this internal covariance
        copy to the standard homoskedastic scale.
''',
    '''        Existing Stage-A inference is computed and stored before this property is
        consulted, so rescaling here cannot change public bse/t/p/CI values.
        Current PanelOLS fits use the standard fixed-effect residual df directly.
        The legacy branch is retained only for compatibility with an already
        materialized result whose metadata explicitly records a legacy-scaled
        homoskedastic covariance.
''',
)

# ---------------------------------------------------------------------------
# 2) Shared rank-deficient inference: HC2/HC3 leverage can be undefined even
#    though the fit itself is valid. Determine rank first and skip an unusable
#    coordinate covariance rather than turning the fit into an exception.
# ---------------------------------------------------------------------------
replace_once(
    "statgpu/panel/_base.py",
    '''        xp = backend.xp
        canonical_cov_type = normalize_covariance_type(cov_type)
        covariance_metadata: dict = {}
        cov_params = ols_covariance(
            X,
            resid,
            cov_type=canonical_cov_type,
            scale=scale,
            df_resid=df_resid,
            cluster=cluster,
            time_ids=time_ids,
            bandwidth=bandwidth,
            kernel=kernel,
            group_debias=group_debias,
            extra_df=extra_df,
            xp=xp,
            allowed=allowed,
            hc1_correction=hc1_correction,
            metadata=covariance_metadata,
        )
        self._covariance_metadata = covariance_metadata
        self._panel_cov_params_raw = np.asarray(
            _to_numpy(cov_params), dtype=np.float64
        )

        diag = xp.diag(cov_params)
        cov_np = self._panel_cov_params_raw
        if not np.all(np.isfinite(cov_np)):
            raise ValueError(
                "covariance contains non-finite values; inference is not numerically valid"
            )

        if fit_rank is None:
            from statgpu.panel._linalg import panel_matrix_rank

            fit_rank = panel_matrix_rank(X, xp)
        fit_rank = int(fit_rank)
        design_columns = int(X.shape[1])
        if fit_rank <= 0 or fit_rank > design_columns:
            raise ValueError(
                "fit_rank must identify a positive subspace no larger than the design"
            )
        rank_deficient = fit_rank < design_columns
''',
    '''        xp = backend.xp
        canonical_cov_type = normalize_covariance_type(cov_type)
        if fit_rank is None:
            from statgpu.panel._linalg import panel_matrix_rank

            fit_rank = panel_matrix_rank(X, xp)
        fit_rank = int(fit_rank)
        design_columns = int(X.shape[1])
        if fit_rank <= 0 or fit_rank > design_columns:
            raise ValueError(
                "fit_rank must identify a positive subspace no larger than the design"
            )
        rank_deficient = fit_rank < design_columns

        covariance_metadata: dict = {}
        # HC2/HC3 divide by (1-h_i).  A rank-deficient design can have h_i=1
        # while still having positive residual df, so the coefficient covariance
        # is undefined even though fitted values remain perfectly well defined.
        # Since coordinate inference is unavailable for every rank-deficient fit
        # anyway, do not let this secondary covariance invalidate the fit itself.
        skip_rank_deficient_hc = rank_deficient and canonical_cov_type in {"hc2", "hc3"}
        if skip_rank_deficient_hc:
            covariance_metadata.update(
                {
                    "covariance": canonical_cov_type,
                    "rank_deficient_covariance_unavailable": True,
                }
            )
            cov_params = None
            self._panel_cov_params_raw = None
        else:
            cov_params = ols_covariance(
                X,
                resid,
                cov_type=canonical_cov_type,
                scale=scale,
                df_resid=df_resid,
                cluster=cluster,
                time_ids=time_ids,
                bandwidth=bandwidth,
                kernel=kernel,
                group_debias=group_debias,
                extra_df=extra_df,
                xp=xp,
                allowed=allowed,
                hc1_correction=hc1_correction,
                metadata=covariance_metadata,
            )
            self._panel_cov_params_raw = np.asarray(
                _to_numpy(cov_params), dtype=np.float64
            )
            cov_np = self._panel_cov_params_raw
            if not np.all(np.isfinite(cov_np)):
                raise ValueError(
                    "covariance contains non-finite values; inference is not numerically valid"
                )
        self._covariance_metadata = covariance_metadata
''',
)
replace_once(
    "statgpu/panel/_base.py",
    '''            ).apply_to(self)
            return cov_params

        diag_np = np.diag(cov_np).astype(np.float64, copy=False)
''',
    '''            ).apply_to(self)
            return cov_params

        diag = xp.diag(cov_params)
        diag_np = np.diag(cov_np).astype(np.float64, copy=False)
''',
)

# ---------------------------------------------------------------------------
# 3) Golden snapshots: coefficients remain unchanged; the two FE bse/df values
#    intentionally move to the standard dummy-regression degrees of freedom.
# ---------------------------------------------------------------------------
replace_once(
    "dev/tests/test_panel_stage_a_golden.py",
    '''    assert_allclose(model.bse_, [0.03123062, 0.03078126], rtol=RTOL, atol=ATOL)
    assert model.df_resid == 31
''',
    '''    # Stage-C correctness revision: public inference counts the full entity
    # nuisance rank, matching the explicit entity-dummy regression.
    assert_allclose(model.bse_, [0.03174686, 0.03129007], rtol=RTOL, atol=ATOL)
    assert model.df_resid == 30
''',
)
replace_once(
    "dev/tests/test_panel_stage_a_golden.py",
    '''    assert_allclose(model.bse_, [0.03275315, 0.03341182], rtol=RTOL, atol=ATOL)
    assert model.df_resid == 27
''',
    '''    # Stage-C correctness revision: connected two-way FE has nuisance rank
    # N + T - 1, so the public residual df is 26 for this fixture.
    assert_allclose(model.bse_, [0.03337708, 0.03404829], rtol=RTOL, atol=ATOL)
    assert model.df_resid == 26
''',
)

# ---------------------------------------------------------------------------
# 4) New executable external/edge regressions.
# ---------------------------------------------------------------------------
test_path = Path("dev/tests/test_panel_stage_c_fresh_df_rank_guard.py")
if test_path.exists():
    raise RuntimeError(f"{test_path}: already exists")
test_path.write_text(r'''"""Fresh review regressions for FE df and rank-deficient HC guards."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

statsmodels = pytest.importorskip("statsmodels.api")

from statgpu.panel import PanelOLS, PooledOLS


def _panel(seed=2026081705):
    rng = np.random.default_rng(seed)
    n_entities, n_times = 9, 5
    entity = np.repeat(np.arange(n_entities), n_times)
    time = np.tile(np.arange(n_times), n_entities)
    X = rng.normal(size=(entity.size, 2))
    alpha = np.repeat(rng.normal(scale=0.4, size=n_entities), n_times)
    tau = np.tile(np.linspace(-0.25, 0.3, n_times), n_entities)
    y = 0.6 * X[:, 0] - 0.35 * X[:, 1] + alpha + 0.2 * tau
    y += rng.normal(scale=0.18, size=entity.size)
    return X, y, entity, time


def _entity_dummy_design(X, entity):
    groups = np.unique(entity)
    dummies = np.column_stack([(entity == group).astype(float) for group in groups])
    return np.column_stack([X, dummies])


def _two_way_dummy_design(X, entity, time):
    entities = np.unique(entity)
    times = np.unique(time)
    entity_dummies = np.column_stack(
        [(entity == group).astype(float) for group in entities]
    )
    time_dummies = np.column_stack(
        [(time == period).astype(float) for period in times[1:]]
    )
    return np.column_stack([X, entity_dummies, time_dummies])


@pytest.mark.parametrize("cov_type", ["nonrobust", "robust"])
def test_one_way_panel_public_inference_matches_explicit_dummy_ols(cov_type):
    X, y, entity, _time = _panel()
    model = PanelOLS(entity_effects=True, cov_type=cov_type).fit(
        X, y, entity_ids=entity
    )
    design = _entity_dummy_design(X, entity)
    reference = statsmodels.OLS(y, design).fit()
    if cov_type == "robust":
        reference = reference.get_robustcov_results(cov_type="HC1")

    k = X.shape[1]
    assert model.df_resid == int(reference.df_resid)
    assert model.fit_statistics_.metadata["public_df_resid_basis"] == "standard"
    assert_allclose(model.coef_, reference.params[:k], rtol=2e-11, atol=2e-12)
    assert_allclose(
        model._panel_cov_params_raw,
        reference.cov_params()[:k, :k],
        rtol=2e-9,
        atol=2e-11,
    )
    assert_allclose(model.bse_, reference.bse[:k], rtol=2e-9, atol=2e-11)


def test_two_way_panel_nonrobust_df_and_covariance_match_explicit_dummies():
    X, y, entity, time = _panel(seed=2026081706)
    model = PanelOLS(
        entity_effects=True,
        time_effects=True,
        cov_type="nonrobust",
    ).fit(X, y, entity_ids=entity, time_ids=time)
    design = _two_way_dummy_design(X, entity, time)
    reference = statsmodels.OLS(y, design).fit()

    k = X.shape[1]
    assert model.df_resid == int(reference.df_resid)
    assert model.fit_statistics_.metadata["diagnostic_df"]["effect_rank"] == (
        len(np.unique(entity)) + len(np.unique(time)) - 1
    )
    assert_allclose(model.coef_, reference.params[:k], rtol=2e-11, atol=2e-12)
    assert_allclose(
        model._panel_cov_params_raw,
        reference.cov_params()[:k, :k],
        rtol=2e-9,
        atol=2e-11,
    )


@pytest.mark.parametrize("cov_type", ["hc2", "hc3"])
def test_rank_deficient_hc_with_unit_leverage_keeps_fit_and_disables_inference(cov_type):
    # PooledOLS adds an intercept.  Together with the duplicated first-row
    # indicator this design has rank 2 < 3 and h_0=1, so HC2/HC3 coordinate
    # covariance is undefined even though the least-squares fitted values exist.
    indicator = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    X = np.column_stack([indicator, 2.0 * indicator])
    y = np.array([1.5, 0.2, -0.1, 0.4, 0.0, 0.3])

    model = PooledOLS(cov_type=cov_type).fit(X, y)
    assert model._coefficient_inference_available is False
    assert model.bse_ is None
    assert model.pvalues_ is None
    assert model.conf_int_ is None
    assert model._panel_cov_params_raw is None
    assert model._covariance_metadata["rank_deficient_covariance_unavailable"] is True
    assert model._covariance_metadata["design_rank"] == 2
    assert model._covariance_metadata["design_columns"] == 3
    pred = model.predict(X)
    assert np.all(np.isfinite(pred))
    with pytest.raises(ValueError, match="rank deficient"):
        model.summary()


def test_one_way_panel_standard_df_torch_cpu_matches_numpy():
    torch = pytest.importorskip("torch")
    X, y, entity, _time = _panel(seed=2026081707)
    expected = PanelOLS(entity_effects=True, cov_type="robust").fit(
        X, y, entity_ids=entity
    )
    actual = PanelOLS(entity_effects=True, cov_type="robust").fit(
        torch.as_tensor(X, dtype=torch.float64),
        torch.as_tensor(y, dtype=torch.float64),
        entity_ids=torch.as_tensor(entity, dtype=torch.int64),
    )
    assert actual.df_resid == expected.df_resid
    assert_allclose(actual.coef_, expected.coef_, rtol=2e-9, atol=2e-11)
    assert_allclose(actual.bse_, expected.bse_, rtol=2e-8, atol=2e-10)
''')

# ---------------------------------------------------------------------------
# 5) Documentation and changelog.
# ---------------------------------------------------------------------------
replace_once(
    "docs/en/panel/panel-ols.md",
    '''where $C$ is the number of connected components in the observed entity-time incidence graph.

## Parameters
''',
    '''where $C$ is the number of connected components in the observed entity-time incidence graph.

The public residual degrees of freedom use this same nuisance-effect rank,

$$
df_{\mathrm{resid}}=n-\operatorname{rank}(Z)-r_F.
$$

This single definition drives the reported `df_resid`, the nonrobust residual-variance scale, the HC1 (`robust`) finite-sample correction, and nonrobust Student-$t$ inference. It is therefore identical to the residual df of the corresponding explicit fixed-effect dummy regression rather than an `N-1`/`T-1` shortcut applied after within transformation.

## Parameters
''',
)
replace_once(
    "docs/cn/panel/panel-ols.md",
    '''其中 $C$ 是观测 entity-time incidence graph 的 connected-component 数。

## Parameters
''',
    '''其中 $C$ 是观测 entity-time incidence graph 的 connected-component 数。

公开的 residual degrees of freedom 使用同一个 nuisance-effect rank：

$$
df_{\mathrm{resid}}=n-\operatorname{rank}(Z)-r_F.
$$

这一统一定义同时决定公开的 `df_resid`、nonrobust residual-variance scale、HC1 (`robust`) finite-sample correction，以及 nonrobust Student-$t$ inference。因此它与对应的显式 fixed-effect dummy regression 的 residual df 完全一致，而不是在 within transformation 后再使用 `N-1`/`T-1` 的简化计数。

## Parameters
''',
)
replace_once(
    "CHANGELOG.md",
    '''### PR #126 — Panel Tier-1 Stage C covariance
- Added HC0/HC2/HC3, robust RandomEffects inference, cluster group debiasing, and Driscoll-Kraay covariance across NumPy, CuPy, and Torch; RandomEffects now also fails closed when the Swamy-Arora between auxiliary regression has no positive residual degrees of freedom.
''',
    '''### PR #126 — Panel Tier-1 Stage C covariance
- Corrected PanelOLS public residual degrees of freedom to count the full identified fixed-effect nuisance rank consistently across nonrobust scale, HC1 correction, Student-t inference, diagnostics, and explicit-dummy references; rank-deficient HC2/HC3 fits now preserve fitted values and fail closed only at coefficient inference when unit leverage makes the coordinate covariance undefined.
- Added HC0/HC2/HC3, robust RandomEffects inference, cluster group debiasing, and Driscoll-Kraay covariance across NumPy, CuPy, and Torch; RandomEffects now also fails closed when the Swamy-Arora between auxiliary regression has no positive residual degrees of freedom.
''',
)
