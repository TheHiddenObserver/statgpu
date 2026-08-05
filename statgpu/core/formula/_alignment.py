"""Alignment helpers for formula-owned side arrays."""

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
