"""Reproduce mixed GAM state after failed basis construction, no state edits."""

import numpy as np

from statgpu.semiparametric import GAM

x = np.linspace(-2, 2, 30)
model = GAM(n_splines=8, lam=1, device="cpu").fit(x, np.sin(x))
print("before:", model.n_features_, len(model.knots_), model.coef_.shape)
x_bad = np.column_stack([
    np.linspace(-1, 1, 30), np.r_[np.zeros(27), np.ones(3)],
])
try:
    model.fit(x_bad, np.arange(30))
except ValueError as error:
    print("refit:", type(error).__name__, str(error))
print("after:", model._fitted, model.n_features_, len(model.knots_), model.coef_.shape)
print("summary:", model.summary())
try:
    print("prediction:", model.predict([[0., .5]]))
except ValueError as error:
    print("prediction:", type(error).__name__, str(error))
