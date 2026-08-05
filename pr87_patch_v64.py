from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one match, found {count}: {old[:180]!r}"
        )
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_once(
    "docs/en/models/elastic-net.md",
    '''**Note on regularization scaling**: With `l1_ratio=0`, `ElasticNet(alpha)` is equivalent to `Ridge(n_samples * alpha)` due to the loss scaling convention.
''',
    '''**Note on regularization scaling**: `ElasticNet` and `Ridge` both use the same average-loss convention. Therefore, with `l1_ratio=0`, `ElasticNet(alpha)` is equivalent to `Ridge(alpha)`; no sample-size rescaling of the public `alpha` is required.
''',
)
replace_once(
    "docs/cn/models/elastic-net.md",
    '''**正则化缩放说明**：当 `l1_ratio=0` 时，`ElasticNet(alpha)` 等价于 `Ridge(n_samples * alpha)`，这是由于损失函数的缩放约定。
''',
    '''**正则化缩放说明**：`ElasticNet` 与 `Ridge` 均采用相同的平均损失尺度。因此当 `l1_ratio=0` 时，`ElasticNet(alpha)` 等价于 `Ridge(alpha)`；公开参数 `alpha` 不需要再乘以样本量。
''',
)

test_path = Path("dev/tests/test_maintenance_024_025.py")
test_text = test_path.read_text(encoding="utf-8")
if "test_elasticnet_zero_l1_ratio_matches_same_alpha_ridge" in test_text:
    raise RuntimeError("v64 test already present")
test_text += r'''


def test_elasticnet_zero_l1_ratio_matches_same_alpha_ridge():
    from statgpu.linear_model import ElasticNet, Ridge

    rng = np.random.default_rng(20260810)
    X = rng.normal(size=(80, 4))
    y = 0.45 + X @ np.array([1.2, -0.8, 0.0, 0.55])
    y = y + rng.normal(scale=0.2, size=X.shape[0])
    alpha = 0.17

    elastic = ElasticNet(
        alpha=alpha,
        l1_ratio=0.0,
        fit_intercept=True,
        solver="fista",
        max_iter=5000,
        tol=1e-10,
        device="cpu",
        compute_inference=False,
    ).fit(X, y)
    ridge = Ridge(
        alpha=alpha,
        fit_intercept=True,
        solver="exact",
        device="cpu",
        compute_inference=False,
    ).fit(X, y)

    np.testing.assert_allclose(
        elastic.coef_, ridge.coef_, rtol=2e-7, atol=2e-8
    )
    np.testing.assert_allclose(
        elastic.intercept_, ridge.intercept_, rtol=2e-7, atol=2e-8
    )
'''
test_path.write_text(test_text, encoding="utf-8")

for changelog, bullet in (
    (
        "CHANGELOG.md",
        "- Corrected ElasticNet/Ridge scaling documentation and added a regression test confirming that `ElasticNet(alpha, l1_ratio=0)` matches `Ridge(alpha)` under the shared average-loss convention.\n",
    ),
    (
        "docs/en/changelog.md",
        "- Corrected ElasticNet/Ridge scaling documentation and added a regression test confirming that `ElasticNet(alpha, l1_ratio=0)` matches `Ridge(alpha)` under the shared average-loss convention.\n",
    ),
    (
        "docs/cn/changelog.md",
        "- 修正 ElasticNet/Ridge 的缩放说明，并补充回归测试确认在共享平均损失尺度下 `ElasticNet(alpha, l1_ratio=0)` 与 `Ridge(alpha)` 一致。\n",
    ),
):
    p = Path(changelog)
    text = p.read_text(encoding="utf-8")
    marker = "# Changelog\n"
    if bullet.strip() not in text:
        text = text.replace(marker, marker + "\n" + bullet, 1)
    p.write_text(text, encoding="utf-8")
