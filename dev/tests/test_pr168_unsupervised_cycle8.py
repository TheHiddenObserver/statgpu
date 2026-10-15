"""Distinct-pair embedding objectives and entropy calibration, NumPy CPU only.

The desired-behavior xfail extends the existing bandwidth-search issue #194.
Only successful acceptance of the specifically reproduced unattainable target
is expected; unrelated exceptions, malformed arrays and invalid densities fail.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import rel_entr, xlogy

from statgpu.unsupervised import TSNE, UMAP

ROOT = Path(__file__).resolve().parents[2]


def _observations():
    return np.array([[0., 0.], [.4, 1.], [1.3, .2],
                     [2.1, 1.9], [3.2, .7], [4., 2.5]])


def _conditional_probabilities(X, perplexity):
    """Independent stable entropy root, using direct pairwise subtraction."""
    n = len(X)
    conditional = np.zeros((n, n))
    for i in range(n):
        other = np.arange(n) != i
        distances = np.sum((X[i] - X[other]) ** 2, axis=1)
        distances -= distances.min()  # Does not change the normalized row.

        def row(beta, distances=distances):
            mass = np.exp(-beta * distances)
            return mass / mass.sum()

        def entropy_difference(beta):
            p = row(beta)
            return -xlogy(p, p).sum() - np.log(perplexity)

        beta = brentq(entropy_difference, 0., 1e4, xtol=1e-13)
        conditional[i, other] = row(beta)
    return conditional


def test_tsne_joint_affinities_match_documented_entropy_and_symmetrization():
    X = _observations()
    perplexity = 3.
    conditional = _conditional_probabilities(X, perplexity)
    np.testing.assert_allclose(np.exp(-xlogy(conditional, conditional).sum(axis=1)),
                               perplexity, rtol=1e-11)
    np.testing.assert_allclose(conditional.sum(axis=1), 1., atol=1e-14)
    expected = (conditional + conditional.T) / (2 * len(X))
    model = TSNE(perplexity=perplexity, max_iter=250, init="random",
                 random_state=13, device="cpu").fit(X)
    actual = model._joint_probabilities(model._get_backend(), X)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(actual, actual.T, atol=1e-14)
    np.testing.assert_allclose(np.diag(actual), 0., atol=1e-299)
    assert actual.sum() == pytest.approx(1., abs=1e-13)
    assert model.embedding_.shape == (len(X), 2)
    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(model.kl_divergence_) and model.kl_divergence_ >= 0


class _UnattainablePerplexityAccepted(Exception):
    """A target above the entropy maximum silently became uniform affinities."""


@pytest.mark.xfail(
    strict=True,
    raises=_UnattainablePerplexityAccepted,
    reason="Issue #194: bandwidth search accepts unattainable perplexity without reporting failure",
)
def test_tsne_rejects_perplexity_above_other_observation_entropy_limit():
    X = _observations()
    model = TSNE(perplexity=len(X) - .5, max_iter=250, init="random",
                 random_state=13, device="cpu")
    try:
        model.fit(X)
    except ValueError as error:
        message = str(error).lower()
        assert "perplexity" in message
        assert any(term in message for term in ("n_samples - 1", "n_samples-1",
                                                "attain", "feasib", "maximum",
                                                "at most 5", "<= 5"))
        assert not model._fitted
        return
    assert model._fitted
    assert model.embedding_.shape == (len(X), 2)
    assert np.isfinite(model.embedding_).all()
    assert np.isfinite(model.kl_divergence_) and model.kl_divergence_ >= 0
    P = model._joint_probabilities(model._get_backend(), X)
    assert np.isfinite(P).all() and np.all(P >= 0)
    assert P.sum() == pytest.approx(1., abs=1e-13)
    off_diagonal = ~np.eye(len(X), dtype=bool)
    expected = 1. / (len(X) * (len(X) - 1))
    np.testing.assert_allclose(P[off_diagonal], expected, rtol=1e-12, atol=1e-14)
    raise _UnattainablePerplexityAccepted(
        "Requested 5.5 neighbors from a five-outcome distribution; accepted uniform affinities"
    )


def test_umap_reference_cross_entropy_excludes_self_pairs():
    # This checks the documented reference objective, not UMAP's known
    # non-gradient force updates. No optimizer-equality claim is made.
    Y = np.array([[0., 0.], [1., .5], [2.5, -.5]])
    weights = np.array([[0., 1., 0.], [1., 0., .4], [0., .4, 0.]])
    distance_sq = np.sum((Y[:, None] - Y[None, :]) ** 2, axis=2)
    affinities = 1. / (1. + .7 * distance_sq ** 1.2)
    terms = rel_entr(weights, affinities) + rel_entr(1 - weights, 1 - affinities)
    rows, cols = np.triu_indices(len(Y), k=1)
    objective = terms[rows, cols].sum()
    assert np.isfinite(objective) and objective >= 0
    # Diagonal terms are infinite: zero graph self-weights versus q_ii=1.
    assert np.isinf(np.diag(terms)).all()
    off_diagonal = ~np.eye(len(Y), dtype=bool)
    assert terms[off_diagonal].sum() == pytest.approx(2 * objective)
    permutation = np.array([2, 0, 1])
    assert terms[np.ix_(permutation, permutation)][rows, cols].sum() == pytest.approx(objective)


def test_umap_attraction_curve_fitting_uses_host_scipy(monkeypatch):
    import scipy.optimize

    original = scipy.optimize.curve_fit
    seen = []

    def record_inputs(function, xdata, ydata, *args, **kwargs):
        assert isinstance(xdata, np.ndarray) and isinstance(ydata, np.ndarray)
        assert np.isfinite(xdata).all() and np.isfinite(ydata).all()
        seen.append((xdata.shape, ydata.shape))
        return original(function, xdata, ydata, *args, **kwargs)

    monkeypatch.setattr(scipy.optimize, "curve_fit", record_inputs)
    model = UMAP(n_neighbors=2, n_epochs=1, init="random", random_state=13,
                 device="cpu").fit(_observations())
    assert len(seen) == 1
    assert seen[0][0] == seen[0][1]
    assert model.embedding_.shape == (6, 2) and np.isfinite(model.embedding_).all()


@pytest.mark.parametrize("language", ("en", "cn"))
def test_embedding_guides_define_distinct_pairs_and_perplexity_limits(language):
    directory = ROOT / f"docs/{language}/unsupervised"
    umap = (directory / "umap.md").read_text()
    tsne = (directory / "tsne.md").read_text()
    reference = (directory / "api-reference.md").read_text()
    assert r"\sum_{i<j}\left[" in umap
    assert r"\sum_{i,j} w_{ij}" not in umap
    assert r"p_{j\mid i}" in tsne and r"p_{ii}=0" in tsne and r"q_{ii}=0" in tsne
    assert r"\operatorname{Perp}_i" in tsne and r"{2n}" in tsne
    assert "0 < perplexity < n" in tsne
    assert "[1,n-1]" in reference
    for page in (umap, reference):
        assert ("attraction-curve fitting" if language == "en" else "吸引曲线拟合") in page
    assert ("tied nearest neighbors" if language == "en" else "并列最近邻") in reference
    assert "attainable" in TSNE.__doc__
    assert "attraction-curve fitting" in UMAP.__doc__
