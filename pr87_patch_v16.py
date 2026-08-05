from pathlib import Path


def replace_once(path, old, new):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"patch anchor missing in {path}: {old[:120]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


alignment = '''"""Alignment helpers for formula-owned side arrays."""

from __future__ import annotations

import numpy as np

from statgpu.backends._validation import check_finite


def align_formula_sample_weight(
    sample_weight,
    *,
    data_length: int,
    retained_rows,
    retained_length: int,
):
    """Align and validate sample weights after formula row filtering.

    The input must be one-dimensional and may describe either the original
    data rows or the rows retained by the formula parser. Validation occurs
    after alignment so non-finite values located only in formula-dropped rows
    do not produce false errors. Torch and CuPy indexing/reductions remain on
    device.
    """
    if sample_weight is None:
        return None

    module = type(sample_weight).__module__
    if module.startswith("pandas"):
        weights = sample_weight.to_numpy()
        module = type(weights).__module__
    else:
        weights = sample_weight

    if getattr(weights, "ndim", None) is None:
        weights = np.asarray(weights)
        module = type(weights).__module__
    if int(weights.ndim) != 1:
        raise ValueError("sample_weight must be one-dimensional")

    n_weights = int(weights.shape[0])
    if n_weights == int(data_length):
        if module.startswith("torch"):
            import torch

            index = torch.as_tensor(
                retained_rows, dtype=torch.long, device=weights.device
            )
            aligned = torch.index_select(weights, 0, index)
        elif module.startswith("cupy"):
            import cupy as cp

            aligned = weights[cp.asarray(retained_rows, dtype=cp.int64)]
        else:
            aligned = np.asarray(weights)[np.asarray(retained_rows, dtype=np.int64)]
    elif n_weights == int(retained_length):
        aligned = weights
    else:
        raise ValueError(
            "sample_weight must match the original data length or the number "
            "of formula rows retained after missing-value filtering"
        )

    check_finite(aligned, name="sample_weight")
    aligned_module = type(aligned).__module__
    if aligned_module.startswith("torch"):
        import torch

        if bool(torch.any(aligned < 0).item()):
            raise ValueError("sample_weight must be non-negative")
        total = float(torch.sum(aligned).item())
    elif aligned_module.startswith("cupy"):
        import cupy as cp

        if bool(cp.any(aligned < 0).item()):
            raise ValueError("sample_weight must be non-negative")
        total = float(cp.sum(aligned).item())
    else:
        aligned_np = np.asarray(aligned)
        if np.any(aligned_np < 0):
            raise ValueError("sample_weight must be non-negative")
        total = float(np.sum(aligned_np))
    if total <= 0.0:
        raise ValueError("sample_weight must have a positive sum")
    return aligned
'''
Path("statgpu/core/formula/_alignment.py").write_text(alignment, encoding="utf-8")

replace_once(
    "statgpu/core/formula/__init__.py",
    '''from ._parser import FormulaParser
from ._design import parse_formula, parse_formula_safe
from ._terms import make_surv_env, _surv
''',
    '''from ._parser import FormulaParser
from ._design import parse_formula, parse_formula_safe
from ._alignment import align_formula_sample_weight
from ._terms import make_surv_env, _surv
''',
)
replace_once(
    "statgpu/core/formula/__init__.py",
    '''    "parse_formula_safe",
    "make_surv_env",
''',
    '''    "parse_formula_safe",
    "align_formula_sample_weight",
    "make_surv_env",
''',
)

replace_once(
    "statgpu/_base.py",
    '''                    formula_owned_pandas = formula_active or (
                        method_name != "fit"
                        and name == "X"
                        and getattr(self, "_design_info", None) is not None
                    )
                    if formula_owned_pandas and type(value).__module__.startswith("pandas"):
''',
    '''                    formula_owned_pandas = formula_active or (
                        method_name != "fit"
                        and name == "X"
                        and getattr(self, "_design_info", None) is not None
                    )
                    formula_owned_side_array = formula_active and name == "sample_weight"
                    if formula_owned_side_array or (
                        formula_owned_pandas
                        and type(value).__module__.startswith("pandas")
                    ):
''',
)
replace_once(
    "statgpu/_base.py",
    '''                        # Current formula calls own all pandas row-alignment semantics.
                        # After a formula fit, only X passed to a prediction-like
                        # method is transformed by stored design_info; direct refits
                        # and side arrays such as y still use the shared finite guard.
''',
    '''                        # Current formula calls own pandas row alignment and
                        # sample-weight alignment. Model-specific formula code checks
                        # the retained side array after Patsy has selected rows.
                        # After a formula fit, only X passed to a prediction-like
                        # method is transformed by stored design_info; direct refits
                        # and unrelated side arrays still use the shared finite guard.
''',
)

replace_once(
    "statgpu/linear_model/wrappers/_linear.py",
    '''            if sample_weight is not None:
                from statgpu.backends import _to_numpy

                weights = np.asarray(_to_numpy(sample_weight), dtype=float)
                if weights.ndim != 1:
                    raise ValueError("sample_weight must be one-dimensional")
                retained_rows = np.asarray(retained_rows, dtype=np.int64)
                if weights.shape[0] == len(data):
                    sample_weight = weights[retained_rows]
                elif weights.shape[0] == len(y_arr):
                    # Already aligned weights are accepted for programmatic use.
                    sample_weight = weights
                else:
                    raise ValueError(
                        "sample_weight must match the original data length or "
                        "the number of formula rows retained after missing-value filtering"
                    )
''',
    '''            if sample_weight is not None:
                from statgpu.core.formula import align_formula_sample_weight

                sample_weight = align_formula_sample_weight(
                    sample_weight,
                    data_length=len(data),
                    retained_rows=retained_rows,
                    retained_length=len(y_arr),
                )
''',
)

replace_once(
    "statgpu/linear_model/_glm_base.py",
    '''            if sample_weight is not None:
                from statgpu.backends import _to_numpy as _formula_to_numpy

                weights = np.asarray(_formula_to_numpy(sample_weight)).reshape(-1)
                if weights.shape[0] == len(data):
                    weights = weights[retained_rows]
                elif weights.shape[0] != X_arr.shape[0]:
                    raise ValueError(
                        "For formula fitting, sample_weight must have length "
                        "len(data) or the number of rows retained by the formula."
                    )
                sample_weight = np.asarray(weights, dtype=np.float64)
''',
    '''            if sample_weight is not None:
                from statgpu.core.formula import align_formula_sample_weight

                sample_weight = align_formula_sample_weight(
                    sample_weight,
                    data_length=len(data),
                    retained_rows=retained_rows,
                    retained_length=X_arr.shape[0],
                )
''',
)

replace_once(
    "statgpu/linear_model/penalized/_fit_mixin.py",
    '''            if sample_weight is not None:
                sw_formula = np.asarray(_to_numpy(sample_weight), dtype=np.float64).reshape(-1)
                row_positions = parser.row_positions
                if sw_formula.shape[0] == len(data):
                    sample_weight = sw_formula[row_positions]
                elif sw_formula.shape[0] == X.shape[0]:
                    sample_weight = sw_formula
                else:
                    raise ValueError(
                        "For formula fitting, sample_weight must have length "
                        "len(data) or the number of rows retained by the formula."
                    )
''',
    '''            if sample_weight is not None:
                from statgpu.core.formula import align_formula_sample_weight

                sample_weight = align_formula_sample_weight(
                    sample_weight,
                    data_length=len(data),
                    retained_rows=parser.row_positions,
                    retained_length=X.shape[0],
                )
''',
)

# Add direct GLM semantic weight validation for every solver/backend.
insert_anchor = '''        # Ensure y is 1D after backend conversion
        if hasattr(y_arr, 'ndim') and y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        self._nobs = X_arr.shape[0]

        family = self._get_family()
'''
insert_new = '''        # Ensure y is 1D after backend conversion
        if hasattr(y_arr, 'ndim') and y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
        self._nobs = X_arr.shape[0]

        if sample_weight is not None:
            sample_weight = self._to_array(sample_weight, backend=backend_name)
            if int(sample_weight.ndim) != 1:
                raise ValueError("sample_weight must be one-dimensional")
            if int(sample_weight.shape[0]) != int(self._nobs):
                raise ValueError("sample_weight must have length n_samples")
            from statgpu.backends._validation import check_finite

            check_finite(sample_weight, name="sample_weight")
            if backend_name == "torch":
                import torch

                if bool(torch.any(sample_weight < 0).item()):
                    raise ValueError("sample_weight must be non-negative")
                weight_sum = float(torch.sum(sample_weight).item())
            elif backend_name == "cupy":
                import cupy as cp

                if bool(cp.any(sample_weight < 0).item()):
                    raise ValueError("sample_weight must be non-negative")
                weight_sum = float(cp.sum(sample_weight).item())
            else:
                if np.any(np.asarray(sample_weight) < 0):
                    raise ValueError("sample_weight must be non-negative")
                weight_sum = float(np.sum(np.asarray(sample_weight)))
            if weight_sum <= 0.0:
                raise ValueError("sample_weight must have a positive sum")

        family = self._get_family()
'''
replace_once("statgpu/linear_model/_glm_base.py", insert_anchor, insert_new)

# Add regression coverage for retained/dropped non-finite values, shape, and
# all formula-capable estimator families.
tests = Path("dev/tests/test_maintenance_024_025.py")
text = tests.read_text(encoding="utf-8")
marker = "# PR87_FORMULA_WEIGHT_SHARED_ALIGNMENT_TESTS"
if marker not in text:
    text += '''

# PR87_FORMULA_WEIGHT_SHARED_ALIGNMENT_TESTS
@pytest.mark.parametrize("kind", ["linear", "glm", "penalized"])
def test_formula_sample_weight_validates_after_row_alignment(kind):
    pd = pytest.importorskip("pandas")
    from statgpu.linear_model import (
        GeneralizedLinearModel,
        LinearRegression,
        PenalizedLinearRegression,
    )

    data = pd.DataFrame(
        {"y": [1.0, 2.0, 3.0, 5.0], "x": [0.0, 1.0, np.nan, 3.0]}
    )
    if kind == "linear":
        factory = lambda: LinearRegression(device="cpu", compute_inference=False)
    elif kind == "glm":
        factory = lambda: GeneralizedLinearModel(
            family="gaussian", solver="irls", C=0.0, device="cpu"
        )
    else:
        factory = lambda: PenalizedLinearRegression(
            loss="squared_error",
            penalty="l1",
            alpha=0.01,
            max_iter=30,
            device="cpu",
            compute_inference=False,
        )

    # Non-finite value belongs only to the row Patsy drops, so it is removed
    # before the retained side array is validated.
    model = factory().fit(
        formula="y ~ x",
        data=data,
        sample_weight=np.array([1.0, 1.0, np.nan, 1.0]),
    )
    assert model is not None

    with pytest.raises(ValueError, match=r"sample_weight.*finite"):
        factory().fit(
            formula="y ~ x",
            data=data,
            sample_weight=pd.Series([1.0, np.nan, 1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        factory().fit(
            formula="y ~ x",
            data=data,
            sample_weight=np.ones((len(data), 1)),
        )
    with pytest.raises(ValueError, match="non-negative"):
        factory().fit(
            formula="y ~ x",
            data=data,
            sample_weight=np.array([1.0, -1.0, 1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="positive sum"):
        factory().fit(
            formula="y ~ x",
            data=data,
            sample_weight=np.zeros(len(data)),
        )


def test_glm_direct_sample_weight_semantic_contract():
    from statgpu.linear_model import GeneralizedLinearModel

    X = np.arange(8.0).reshape(4, 2)
    y = np.arange(4.0)
    factory = lambda: GeneralizedLinearModel(
        family="gaussian", solver="irls", C=0.0, device="cpu"
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        factory().fit(X, y, sample_weight=np.ones((4, 1)))
    with pytest.raises(ValueError, match="non-negative"):
        factory().fit(X, y, sample_weight=np.array([1.0, 1.0, -1.0, 1.0]))
    with pytest.raises(ValueError, match="positive sum"):
        factory().fit(X, y, sample_weight=np.zeros(4))
'''
    tests.write_text(text, encoding="utf-8")
