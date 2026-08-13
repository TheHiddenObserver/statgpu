from pathlib import Path


def replace_exact(path, old, new, *, expected=1, replace_count=None):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    actual = text.count(old)
    if actual != expected:
        raise SystemExit(
            f"{path}: expected {expected} occurrence(s), found {actual}: {old!r}"
        )
    limit = expected if replace_count is None else int(replace_count)
    p.write_text(text.replace(old, new, limit), encoding="utf-8")


# PanelOLS: expose and validate two-way projection convergence controls.
path = "statgpu/panel/_fixed_effects.py"
replace_exact(
    path,
    """        group_debias: bool = False,\n    ):""",
    """        group_debias: bool = False,\n        demean_max_iter: int = 1_000_000,\n        demean_tol: float = 1e-10,\n    ):""",
)
replace_exact(
    path,
    """        self.group_debias = group_debias\n        allowed = {""",
    """        self.group_debias = group_debias\n        if isinstance(demean_max_iter, (bool, np.bool_)) or not isinstance(\n            demean_max_iter, (int, np.integer)\n        ) or int(demean_max_iter) <= 0:\n            raise ValueError(\"demean_max_iter must be a positive integer\")\n        if not np.isfinite(float(demean_tol)) or float(demean_tol) <= 0.0:\n            raise ValueError(\"demean_tol must be finite and positive\")\n        self.demean_max_iter = int(demean_max_iter)\n        self.demean_tol = float(demean_tol)\n        allowed = {""",
)
replace_exact(
    path,
    """                time_ids=time_arr if self.time_effects else None,\n                xp=xp,\n            )""",
    """                time_ids=time_arr if self.time_effects else None,\n                xp=xp,\n                max_iter=self.demean_max_iter,\n                tol=self.demean_tol,\n            )""",
)
replace_exact(
    path,
    """            ent_effects_dev, time_effects_dev = _recover_two_way_effects(\n                resid_centered,\n                entity_arr,\n                time_arr,\n                xp,\n            )""",
    """            ent_effects_dev, time_effects_dev = _recover_two_way_effects(\n                resid_centered,\n                entity_arr,\n                time_arr,\n                xp,\n                max_iter=self.demean_max_iter,\n                tol=self.demean_tol,\n            )""",
)
replace_exact(
    path,
    """    def set_params(self, **params):\n        \"\"\"Delegate parameter updates to the shared estimator contract.\"\"\"\n        return super().set_params(**params)""",
    """    def set_params(self, **params):\n        \"\"\"Validate FE convergence controls and delegate estimator updates.\"\"\"\n        if \"demean_max_iter\" in params:\n            value = params[\"demean_max_iter\"]\n            if isinstance(value, (bool, np.bool_)) or not isinstance(\n                value, (int, np.integer)\n            ) or int(value) <= 0:\n                raise ValueError(\"demean_max_iter must be a positive integer\")\n        if \"demean_tol\" in params:\n            value = params[\"demean_tol\"]\n            if not np.isfinite(float(value)) or float(value) <= 0.0:\n                raise ValueError(\"demean_tol must be finite and positive\")\n        return super().set_params(**params)""",
)

# Replace the saturated path fixture by a sparse connected graph with cycles,
# so slow convergence is exercised while the within residual space is nonzero.
test_path = "dev/tests/test_panel_stage_c_final_review_fixes.py"
old_weak = """def test_weakly_connected_two_way_panel_converges_beyond_legacy_100_iterations():\n    rng = np.random.default_rng(20260818)\n    n_entities = 20\n    entity = np.repeat(np.arange(n_entities), 2)\n    time = np.column_stack(\n        [np.arange(n_entities), np.arange(1, n_entities + 1)]\n    ).ravel()\n    X = rng.normal(size=(entity.size, 2))\n    y = rng.normal(size=entity.size)\n\n    with pytest.raises(RuntimeError, match=\"did not converge\"):\n        demean_variables(\n            y,\n            X,\n            entity,\n            time,\n            xp=np,\n            max_iter=100,\n            tol=1e-10,\n        )\n\n    y_d, X_d = demean_variables(\n        y,\n        X,\n        entity,\n        time,\n        xp=np,\n        max_iter=50_000,\n        tol=1e-10,\n    )\n    _assert_two_way_means_zero(y_d, entity, time, atol=3e-10)\n    for column in range(X_d.shape[1]):\n        _assert_two_way_means_zero(X_d[:, column], entity, time, atol=3e-10)\n\n\n"""
new_weak = """def test_weakly_connected_two_way_panel_converges_beyond_legacy_100_iterations():\n    rng = np.random.default_rng(20260818)\n    n_entities = 30\n    entity = np.repeat(np.arange(n_entities), 3)\n    time = np.column_stack(\n        [\n            np.arange(n_entities),\n            np.arange(1, n_entities + 1),\n            np.arange(2, n_entities + 2),\n        ]\n    ).ravel()\n    X = rng.normal(size=(entity.size, 2))\n    y = rng.normal(size=entity.size)\n\n    with pytest.raises(RuntimeError, match=\"did not converge\"):\n        demean_variables(\n            y,\n            X,\n            entity,\n            time,\n            xp=np,\n            max_iter=100,\n            tol=1e-10,\n        )\n\n    y_d, X_d = demean_variables(\n        y,\n        X,\n        entity,\n        time,\n        xp=np,\n        max_iter=10_000,\n        tol=1e-10,\n    )\n    _assert_two_way_means_zero(y_d, entity, time, atol=3e-10)\n    for column in range(X_d.shape[1]):\n        _assert_two_way_means_zero(X_d[:, column], entity, time, atol=3e-10)\n\n\n"""
replace_exact(test_path, old_weak, new_weak)
marker = "\ndef test_numerically_absorbed_two_way_direction_terminates_without_relative_zero_trap():\n"
addition = """
def test_panel_exposes_fail_closed_two_way_convergence_controls():
    rng = np.random.default_rng(202608181)
    n_entities = 30
    entity = np.repeat(np.arange(n_entities), 3)
    time = np.column_stack(
        [
            np.arange(n_entities),
            np.arange(1, n_entities + 1),
            np.arange(2, n_entities + 2),
        ]
    ).ravel()
    X = rng.normal(size=(entity.size, 2))
    y = 0.5 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(scale=0.2, size=entity.size)

    with pytest.raises(RuntimeError, match="did not converge"):
        PanelOLS(
            entity_effects=True,
            time_effects=True,
            demean_max_iter=100,
            demean_tol=1e-10,
        ).fit(X, y, entity_ids=entity, time_ids=time)

    model = PanelOLS(
        entity_effects=True,
        time_effects=True,
        demean_max_iter=10_000,
        demean_tol=1e-10,
    ).fit(X, y, entity_ids=entity, time_ids=time)
    assert model.demean_max_iter == 10_000
    assert model.demean_tol == 1e-10
    assert model.get_params()["demean_max_iter"] == 10_000
    assert model.get_params()["demean_tol"] == 1e-10

    with pytest.raises(ValueError, match="positive integer"):
        PanelOLS(demean_max_iter=0)
    with pytest.raises(ValueError, match="finite and positive"):
        PanelOLS(demean_tol=0.0)
    with pytest.raises(ValueError, match="positive integer"):
        PanelOLS().set_params(demean_max_iter=False)

"""
replace_exact(test_path, marker, addition + marker)

# Stable model documentation: expose numerical controls and prediction boundary.
en_model = "docs/en/models/panel.md"
replace_exact(en_model, "> Last updated: 2026-08-12", "> Last updated: 2026-08-13")
replace_exact(
    en_model,
    """PanelOLS(\n    entity_effects=False,\n    time_effects=False,\n    cov_type=\"nonrobust\",  # robust/hc0/hc1/hc2/hc3/clustered/dk also supported\n    bandwidth=None,\n    kernel=\"bartlett\",\n    group_debias=False,\n    device=\"auto\",\n)""",
    """PanelOLS(\n    entity_effects=False,\n    time_effects=False,\n    cov_type=\"nonrobust\",  # robust/hc0/hc1/hc2/hc3/clustered/dk also supported\n    bandwidth=None,\n    kernel=\"bartlett\",\n    group_debias=False,\n    demean_max_iter=1_000_000,\n    demean_tol=1e-10,\n    device=\"auto\",\n)""",
)
replace_exact(
    en_model,
    """Formula input can also request effects through the existing pipe syntax, for example `\"y ~ x1 + x2 | entity\"`. Formula row filtering aligns side arrays to the retained estimation sample.""",
    """Formula input can also request effects through the existing pipe syntax, for example `\"y ~ x1 + x2 | entity\"`. Formula row filtering aligns side arrays to the retained estimation sample. For two-way fixed effects, `demean_max_iter` and `demean_tol` control the fail-closed backend-native projection solver; convergence is checked against residual entity and time means rather than only adjacent-iterate changes. Group codes are factorized once and reused across iterations. Fitted two-way effects are recovered jointly on unbalanced panels. If the observed entity-time incidence graph is disconnected, a prediction combining known entity and time labels from different components is not identified and raises instead of adding arbitrary component normalizations.""",
)

cn_model = "docs/cn/models/panel.md"
replace_exact(cn_model, "> 最后更新：2026-08-12", "> 最后更新：2026-08-13")
replace_exact(
    cn_model,
    """PanelOLS(\n    entity_effects=False,\n    time_effects=False,\n    cov_type=\"nonrobust\",  # 同时支持 robust/hc0/hc1/hc2/hc3/clustered/dk\n    bandwidth=None,\n    kernel=\"bartlett\",\n    group_debias=False,\n    device=\"auto\",\n)""",
    """PanelOLS(\n    entity_effects=False,\n    time_effects=False,\n    cov_type=\"nonrobust\",  # 同时支持 robust/hc0/hc1/hc2/hc3/clustered/dk\n    bandwidth=None,\n    kernel=\"bartlett\",\n    group_debias=False,\n    demean_max_iter=1_000_000,\n    demean_tol=1e-10,\n    device=\"auto\",\n)""",
)
replace_exact(
    cn_model,
    """formula 输入也可使用已有 pipe syntax，例如 `\"y ~ x1 + x2 | entity\"`。formula 的 missing-row filtering 会同步对齐 observation-level side arrays。""",
    """formula 输入也可使用已有 pipe syntax，例如 `\"y ~ x1 + x2 | entity\"`。formula 的 missing-row filtering 会同步对齐 observation-level side arrays。对于 two-way fixed effects，`demean_max_iter` 与 `demean_tol` 控制 fail-closed、backend-native 的 projection solver；收敛判据直接检查残余 entity/time group mean，而不是只比较相邻 iterate。group code 只在迭代前 factorize 一次并反复复用。unbalanced panel 的 two-way fixed effects 会联合恢复；若已观察 entity-time incidence graph 不连通，则来自不同 component 的已知 entity/time label 组合不可识别，prediction 会明确报错，而不会把任意 component normalization 相加。""",
)

# Detailed changelog is the correct home for lifecycle/review-level implementation detail.
en_change = "docs/en/changelog.md"
replace_exact(en_change, "> Last updated: 2026-08-12", "> Last updated: 2026-08-13")
replace_exact(
    en_change,
    """The implementation also hardens two-way fixed-effect convergence, including scaling the alternating-projection stopping rule in the transformed fit space so removable entity-level offsets cannot cause premature convergence; rank-deficient coefficient identifiability; backend-native `PanelOLS.predict()`; `FirstDifferenceOLS` duplicate/time semantics; stable HC2/HC3 leverage; metadata alignment; CuPy scatter-add; RandomEffects formula intercept/name behavior; and quadratic-spectral weights. External definitions are checked against pinned `statsmodels`, `linearmodels`, and R `sandwich`/`plm` references.""",
    """The implementation also hardens two-way fixed-effect convergence and prediction. Entity/time projection metadata is factorized once and reused on the selected backend; convergence checks residual means for both effect dimensions, uses a scale-aware roundoff floor for numerically absorbed directions, and exposes fail-closed `demean_max_iter`/`demean_tol` controls for weakly connected panels. Two-way fixed effects are recovered jointly for unbalanced prediction, while known entity/time labels from different disconnected incidence components are rejected as unidentified. Prediction no longer guesses that every one-column-short matrix omitted an intercept, and an explicitly fitted non-unit constant is restored by value and position only on the compatible path. The same review also strengthens rank-deficient coefficient identifiability, `FirstDifferenceOLS` duplicate/time semantics, HC2/HC3 leverage stability, metadata alignment, CuPy scatter-add, RandomEffects formula intercept/name behavior, and quadratic-spectral weights. External definitions are checked against pinned `statsmodels==0.14.6`, `linearmodels==7.0`, R `plm==2.6-7`, and R `sandwich==3.1-3` references.""",
)

cn_change = "docs/cn/changelog.md"
replace_exact(cn_change, "> 最后更新：2026-08-12", "> 最后更新：2026-08-13")
replace_exact(
    cn_change,
    """本次还强化了 two-way fixed-effect 收敛，其中 alternating-projection stopping rule 改为按 transformed fit space 缩放，避免已被消除的 entity-level offset 造成过早收敛；同时强化 rank-deficient coefficient identifiability、backend-native `PanelOLS.predict()`、`FirstDifferenceOLS` duplicate/time 语义、HC2/HC3 leverage 稳定性、metadata alignment、CuPy scatter-add、RandomEffects formula intercept/name 行为以及 quadratic-spectral weight。外部定义继续对齐固定版本的 `statsmodels`、`linearmodels` 和 R `sandwich`/`plm`。""",
    """本次还系统强化了 two-way fixed-effect 的收敛与 prediction。entity/time projection metadata 只在迭代前 factorize 一次，并在所选 backend 上复用；收敛判据直接检查两个 effect 维度的 residual group mean，对数值上被固定效应完全吸收的方向使用 scale-aware roundoff floor，同时公开 fail-closed 的 `demean_max_iter`/`demean_tol` 控制以处理弱连通 panel。unbalanced panel 的 two-way fixed effects 改为联合恢复；若已知 entity/time label 分属 disconnected incidence graph 的不同 component，则该预测不可识别并明确报错。prediction 也不再把任意少一列的矩阵猜成“省略 intercept”，只有与拟合设计一致时才按原位置和原数值恢复显式 non-unit constant。其余强化还包括 rank-deficient coefficient identifiability、`FirstDifferenceOLS` duplicate/time 语义、HC2/HC3 leverage 稳定性、metadata alignment、CuPy scatter-add、RandomEffects formula intercept/name 行为以及 quadratic-spectral weight。外部定义固定对齐 `statsmodels==0.14.6`、`linearmodels==7.0`、R `plm==2.6-7` 与 R `sandwich==3.1-3`。""",
)
