# OAS: Gaussian-Based Shrinkage for Small Samples

> Language: English
> Last updated: 2026-10-12
> Switch: [Chinese](../../cn/models/covariance-oas.md)

## Can a Gaussian assumption help when observations are scarce?

With many features and few observations, an empirical covariance can be noisy
even when every observation is valid. `OAS` (Oracle Approximating Shrinkage)
uses a Gaussian-based derivation to estimate how strongly to pull that covariance
toward a scaled identity. It is a useful candidate when approximately Gaussian
observations are plausible and the sample is small relative to the dimension.

“Oracle” refers to an ideal shrinkage rule that would require the unknown
population covariance. OAS approximates that rule using sample quantities;
it does not know the true covariance or guarantee the best result on your data.
Start with the [covariance overview](covariance.md) for the wider method choice.

## The estimate and the formula actually used

As with [Ledoit–Wolf](covariance-ledoit-wolf.md), the output is

$$
\widehat\Sigma=(1-\alpha)S+\alpha\mu I,
\qquad S=\frac{1}{n}\sum_{i=1}^n z_i z_i^\top,
\qquad \mu=\frac{\operatorname{tr}(S)}{p}.
$$

By default $z_i=x_i-\bar x$. With `assume_centered=True`, $z_i=x_i$ and the
location is fixed to zero. Define the mean of the squared matrix entries as
$q=\lVert S\rVert_F^2/p^2$. This implementation computes

$$
\alpha=\operatorname{clip}\left(
\frac{q+\mu^2}{(n+1)(q-\mu^2/p)},0,1\right).
$$

If the denominator is nonpositive, it uses $\alpha=1$.

This is the **simplified scikit-learn OAS formula**. It omits the $2/p$
correction terms from equation (23) of Chen et al. (2010). Therefore directly
substituting the full paper formula, particularly for small $p$, need not
reproduce `shrinkage_`. The distinction is also explained in the
[scikit-learn OAS notes](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html#notes).

## Which alternative should you compare?

- [Ledoit–Wolf](covariance-ledoit-wolf.md) estimates shrinkage without the
  Gaussian-specific OAS derivation and is a useful automatic baseline.
- [ShrunkCovariance](covariance-shrunk.md) gives direct control of the strength,
  including selection using a downstream validation criterion.
- [EmpiricalCovariance](covariance-empirical.md) shows what happens without
  shrinkage when the data are sufficiently informative.

OAS regularizes sampling noise. It is not a robust estimator for contaminated
data, and its precision matrix is not designed to be sparse.

## CPU example: compare two automatic choices

This complete example uses 25 training observations for 20 correlated Gaussian
features. It both checks the simplified formula and compares OAS with
Ledoit–Wolf on 300 new observations. Test rows are not used to fit either model
or determine its shrinkage strength.

<!-- example: covariance-oas -->
```python
import numpy as np
from statgpu.covariance import LedoitWolf, OAS

rng = np.random.default_rng(42)
p = 20
population = 0.6 ** np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])
L = np.linalg.cholesky(population)
X_train = rng.normal(size=(25, p)) @ L.T
X_test = rng.normal(size=(300, p)) @ L.T
```

Fit both automatic estimators on the training sample.

```python
oas = OAS(device="cpu").fit(X_train)
lw = LedoitWolf(device="cpu").fit(X_train)
```

Reconstruct the simplified OAS intensity from the empirical covariance.

```python
S = np.cov(X_train, rowvar=False, ddof=0)
mu = np.trace(S) / p
q = np.mean(S * S)
denominator = (len(X_train) + 1) * (q - mu * mu / p)
alpha = (1.0 if denominator <= 0 else
         np.clip((q + mu * mu) / denominator, 0.0, 1.0))
```

Check the formula and compare the fitted models on new rows.

```python
print(f"shrinkage: OAS={oas.shrinkage_:.3f}, Ledoit-Wolf={lw.shrinkage_:.3f}")
print("formula agrees:", np.isclose(oas.shrinkage_, alpha))
print(f"held-out scores: OAS={oas.score(X_test):.3f},"
      f" Ledoit-Wolf={lw.score(X_test):.3f}")
print(f"smallest OAS eigenvalue: {np.linalg.eigvalsh(oas.covariance_)[0]:.3f}")
```
<!-- example-end: covariance-oas -->

Output:

```text
shrinkage: OAS=0.448, Ledoit-Wolf=0.400
formula agrees: True
held-out scores: OAS=-26.540, Ledoit-Wolf=-26.557
smallest OAS eigenvalue: 0.441
```

OAS shrinks a little more in this sample. The formula check uses divisor $n$
through `ddof=0`; using NumPy's default $n-1$ normalization would compare a
different covariance scale. The positive minimum eigenvalue indicates that the
shrunk covariance can be inverted without a zero-variance direction here.

Both scores are average Gaussian log likelihoods; higher is better on the same
observations and scale. OAS is only slightly better in this example. That small
difference is not evidence that it will always beat Ledoit–Wolf. If you use
these scores to select the method, reserve a separate final test set.

## Parameters, assumptions, and limits

OAS has no user-set shrinkage strength or CV search. `shrinkage_` is an output,
while `assume_centered` controls whether the training mean is estimated. Keep
its default `False` unless zero is the intended center. Preprocessing and
feature selection must also be learned from training data only.

The target assumes equal variances in the chosen units. Standardizing features
changes that target and thus the statistical problem; use training-derived
scales if you choose to do so. Strong non-Gaussian tails, outliers, or a very
different covariance structure can make the OAS assumptions unattractive.

A strength of `1` means the estimate uses $\mu I$ completely, not that the data
prove independence. Positive shrinkage only guarantees positive eigenvalues
when $\mu>0$; all-zero variation remains degenerate. The fit requires at least
two finite rows and one feature. For robust or sparse alternatives, see the
[overview](covariance.md); for shared precision behavior, see the
[API method reference](../reference/covariance-api.md#fit-score-predict-and-mahalanobis).

## API and references

The [complete covariance API](../reference/covariance-api.md) documents all
parameters, attributes, methods, and input requirements; see its
[device and dtype section](../reference/covariance-api.md#backend-device-and-dtype)
for device selection.

- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010).
  [Shrinkage algorithms for MMSE covariance estimation](https://arxiv.org/abs/0907.4698).
- [scikit-learn: OAS](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html),
  especially the distinction from the original paper's formula.
