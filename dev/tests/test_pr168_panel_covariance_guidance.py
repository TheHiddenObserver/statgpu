"""Panel covariance guidance: model support and the executable two-way example."""
from pathlib import Path

import numpy as np
import pytest

from doc_examples import example_code, run_example
from statgpu.panel import (BetweenOLS, FamaMacBeth, FirstDifferenceOLS,
                           PanelOLS, PooledOLS, RandomEffects)

ROOT = Path(__file__).resolve().parents[2]


def read(language):
    return (ROOT / 'docs' / language / 'panel/covariance.md').read_text()


def test_two_way_example_has_bilingual_parity_and_required_theory_first():
    en, cn = read('en'), read('cn')
    assert example_code(en, 'covariance-two-way') == example_code(cn, 'covariance-two-way')
    for text in (en, cn):
        example = text.index('<!-- example: covariance-two-way -->')
        assert text.index(r'\widehat V_{1,2}') < example
        assert text.index('Stock') < example
        assert text.index('id="ordered-sequence-hac"') < example
        assert text.index('id="coefficient-series-conditions"') < example
        assert 'np.column_stack([entity_ids, time_ids])' in text
        assert 'https://www.princeton.edu/~mwatson/papers/Stock_Watson_Ecta_2008.pdf' in text
        assert 'https://doi.org/10.1162/003465398557825' in text


@pytest.mark.parametrize('language', ['en', 'cn'])
def test_documented_two_way_example_uses_marginal_partitions(language):
    ns = run_example(read(language), 'covariance-two-way')
    X, y, entity, time = [ns[k] for k in ('X', 'y', 'entity_ids', 'time_ids')]
    np.testing.assert_array_equal(ns['cluster'][:, 0], entity)
    np.testing.assert_array_equal(ns['cluster'][:, 1], time)
    model = ns['model']
    assert model.entity_effects and not model.time_effects
    assert np.isfinite(model.bse_).all() and (model.bse_ > 0).all()
    assert not hasattr(model, 'cov_params_')
    # Build the within scores independently and apply inclusion-exclusion.
    n, t = ns['n_entities'], ns['n_periods']
    Z = X - X.reshape(n, t, 1).mean(axis=1)[entity]
    w = y - y.reshape(n, t).mean(axis=1)[entity]
    beta = np.linalg.lstsq(Z, w, rcond=None)[0]
    score = Z * (w - Z @ beta)[:, None]
    bread = np.linalg.inv(Z.T @ Z)
    def component(labels):
        sums = np.array([score[labels == group].sum(axis=0)
                         for group in np.unique(labels)])
        return bread @ sums.T @ sums @ bread
    expected = component(entity) + component(time) - component(entity * t + time)
    np.testing.assert_allclose(model.bse_ ** 2, np.diag(expected), rtol=1e-10)
    np.testing.assert_allclose(model.coef_, beta, rtol=1e-10)


@pytest.mark.parametrize('cls', [PooledOLS, PanelOLS, RandomEffects, BetweenOLS, FirstDifferenceOLS])
@pytest.mark.parametrize('cov', ['nonrobust', 'robust', 'hc0', 'hc1', 'hc2', 'hc3'])
def test_documented_residual_models_accept_hc_names(cls, cov):
    cls(cov_type=cov, device='cpu')


@pytest.mark.parametrize('cls', [PooledOLS, PanelOLS, RandomEffects])
@pytest.mark.parametrize('cov', ['clustered', 'driscoll-kraay', 'dk', 'kernel'])
def test_documented_cluster_and_dk_support(cls, cov):
    cls(cov_type=cov, device='cpu')


@pytest.mark.parametrize('cls', [PanelOLS, RandomEffects, BetweenOLS, FirstDifferenceOLS])
def test_pooled_only_sequence_hac(cls):
    PooledOLS(cov_type='hac', device='cpu')
    with pytest.raises(ValueError):
        cls(cov_type='hac', device='cpu')


@pytest.mark.parametrize('cls', [BetweenOLS, FirstDifferenceOLS])
@pytest.mark.parametrize('cov', ['clustered', 'driscoll-kraay'])
def test_between_and_fd_do_not_offer_dependence_covariance(cls, cov):
    with pytest.raises(ValueError):
        cls(cov_type=cov, device='cpu')


def test_fama_macbeth_covariance_is_distinct():
    for cov in ('nonrobust', 'newey-west'):
        FamaMacBeth(cov_type=cov, device='cpu')
    for cov in ('robust', 'clustered', 'hac', 'driscoll-kraay'):
        with pytest.raises(ValueError):
            FamaMacBeth(cov_type=cov, device='cpu')
