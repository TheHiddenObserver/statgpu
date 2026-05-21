import numpy as np

from statgpu.glm_core._family import Gamma, InversePowerLink, LogLink


def test_gamma_log_link_irls_matches_closed_form():
    family = Gamma(link=LogLink())
    mu = np.array([0.5, 1.0, 2.0])
    y = np.array([0.4, 1.2, 1.6])
    eta = family.link.link(mu)

    w = family.irls_weights(mu, y)
    z = family.irls_working_response(mu, y, eta)

    expected_w = np.ones_like(mu)
    expected_z = eta + (y - mu) / mu

    assert np.allclose(w, expected_w)
    assert np.allclose(z, expected_z)


def test_gamma_inverse_power_link_irls_uses_link_derivative():
    family = Gamma(link=InversePowerLink())
    mu = np.array([0.5, 1.0, 2.0])
    y = np.array([0.4, 1.2, 1.6])
    eta = family.link.link(mu)

    w = family.irls_weights(mu, y)
    z = family.irls_working_response(mu, y, eta)

    # Gamma variance: V(mu)=mu^2, inverse-power derivative: g'(mu)=-1/mu^2
    expected_w = 1.0 / (mu * mu)
    expected_z = eta - (y - mu) / (mu * mu)

    assert np.allclose(w, expected_w)
    assert np.allclose(z, expected_z)
