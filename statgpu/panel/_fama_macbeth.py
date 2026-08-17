"""Fama-MacBeth two-pass regression with backend-native core linear algebra."""

from __future__ import annotations

__all__ = ["FamaMacBeth"]

from typing import Optional, Union

import numpy as np

from statgpu._config import Device
from statgpu.backends import (
    _get_xp,
    _to_float_scalar,
    _to_numpy,
    xp_asarray,
    xp_ones,
)
from statgpu.covariance._empirical import _detect_backend
from statgpu.inference._reference_distribution import two_sided_reference_inference
from statgpu.panel._base import BasePanelModel
from statgpu.panel._linalg import (
    panel_lstsq,
    panel_lstsq_batched,
    panel_lstsq_deferred_rank,
    panel_lstsq_gram_certified_batched,
)
from statgpu.panel._utils import factorize_panel_labels, factorize_panel_metadata


def _stack(values, xp, axis=0):
    return xp.stack(values, dim=axis) if xp.__name__ == "torch" else xp.stack(values, axis=axis)


def _index_array(indices, xp, ref):
    return xp_asarray(
        np.asarray(indices, dtype=np.int64), dtype=xp.int64, xp=xp, ref_arr=ref
    )


def _finite_all(x, xp):
    return bool(_to_float_scalar(xp.all(xp.isfinite(x))))


def _eligible_period_codes(counts, *, min_obs_per_period, k):
    return [
        int(code)
        for code, n_t in enumerate(counts)
        if int(n_t) >= int(min_obs_per_period) and int(n_t) >= int(k)
    ]


def _group_period_rows(X_design, y_arr, time_codes, counts, xp):
    """Return chronology-grouped rows plus period offsets with at most one gather."""
    grouped = bool(
        len(time_codes) < 2 or np.all(time_codes[:-1] <= time_codes[1:])
    )
    if grouped:
        X_grouped = X_design
        y_grouped = y_arr
    else:
        order = _index_array(np.argsort(time_codes, kind="stable"), xp, X_design)
        X_grouped = X_design[order]
        y_grouped = y_arr[order]

    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts, dtype=np.int64)
    return X_grouped, y_grouped, offsets


def _period_batch(X_grouped, y_grouped, offsets, codes, n_t, k, xp):
    contiguous_codes = codes == list(range(codes[0], codes[0] + len(codes)))
    if contiguous_codes:
        start = int(offsets[codes[0]])
        stop = int(offsets[codes[-1] + 1])
        return (
            X_grouped[start:stop].reshape(len(codes), n_t, k),
            y_grouped[start:stop].reshape(len(codes), n_t),
        )
    return (
        _stack(
            [
                X_grouped[int(offsets[code]) : int(offsets[code + 1])]
                for code in codes
            ],
            xp,
            axis=0,
        ),
        _stack(
            [
                y_grouped[int(offsets[code]) : int(offsets[code + 1])]
                for code in codes
            ],
            xp,
            axis=0,
        ),
    )


def _gpu_certified_period_betas(
    X_design,
    y_arr,
    time_codes,
    counts,
    time_labels,
    *,
    min_obs_per_period,
    backend_name,
    xp,
):
    """Solve GPU periods with a conservative Gram fast path and SVD fallback.

    Equal-sized periods are grouped into one batch.  A batch first receives a
    backend-native Gram-spectrum certificate.  Clearly well-conditioned periods
    use the batched normal-equation solve; every uncertified period falls back to
    the existing SVD policy, so rank-boundary and ill-conditioned behavior remain
    governed by the same singular-value cutoff as the serial reference.
    """
    k = int(X_design.shape[1])
    eligible_codes = _eligible_period_codes(
        counts,
        min_obs_per_period=min_obs_per_period,
        k=k,
    )
    if not eligible_codes:
        return None, 0, 0, 0

    X_grouped, y_grouped, offsets = _group_period_rows(
        X_design, y_arr, time_codes, counts, xp
    )
    buckets = {}
    for code in eligible_codes:
        buckets.setdefault(int(counts[code]), []).append(code)

    beta_by_code = {}
    rank_by_code = {}
    rank_syncs = 0
    svd_fallbacks = 0
    for n_t, codes in buckets.items():
        X_batch, y_batch = _period_batch(
            X_grouped, y_grouped, offsets, codes, n_t, k, xp
        )
        candidate, certified_backend = panel_lstsq_gram_certified_batched(
            X_batch,
            y_batch,
            xp,
        )
        certified = np.asarray(
            _to_numpy(certified_backend), dtype=bool
        ).reshape(-1)
        rank_syncs += 1

        for position, code in enumerate(codes):
            if bool(certified[position]):
                beta_by_code[code] = candidate[position]
                rank_by_code[code] = k

        unsafe_positions = np.flatnonzero(~certified)
        if unsafe_positions.size == 0:
            continue
        svd_fallbacks += int(unsafe_positions.size)

        if backend_name == "torch":
            unsafe_index = _index_array(unsafe_positions, xp, X_batch)
            fallback_betas, fallback_ranks_backend = panel_lstsq_batched(
                X_batch[unsafe_index],
                y_batch[unsafe_index],
                xp,
            )
            fallback_ranks = np.asarray(
                _to_numpy(fallback_ranks_backend), dtype=np.int64
            ).reshape(-1)
            rank_syncs += 1
            for fallback_position, batch_position in enumerate(unsafe_positions):
                code = codes[int(batch_position)]
                beta_by_code[code] = fallback_betas[fallback_position]
                rank_by_code[code] = int(fallback_ranks[fallback_position])
        else:
            fallback_betas = []
            fallback_rank_scalars = []
            for batch_position in unsafe_positions:
                position = int(batch_position)
                beta_t, rank_backend = panel_lstsq_deferred_rank(
                    X_batch[position],
                    y_batch[position],
                    xp,
                )
                fallback_betas.append(beta_t)
                fallback_rank_scalars.append(rank_backend)
            fallback_ranks = np.asarray(
                _to_numpy(_stack(fallback_rank_scalars, xp, axis=0)),
                dtype=np.int64,
            ).reshape(-1)
            rank_syncs += 1
            for fallback_position, batch_position in enumerate(unsafe_positions):
                code = codes[int(batch_position)]
                beta_by_code[code] = fallback_betas[fallback_position]
                rank_by_code[code] = int(fallback_ranks[fallback_position])

    for code in eligible_codes:
        rank = rank_by_code[code]
        if rank < k:
            raise ValueError(
                "FamaMacBeth requires full column rank in every retained period; "
                f"retained time period {time_labels[code]!r} is rank deficient "
                f"(rank={rank}, columns={k})"
            )

    return (
        _stack([beta_by_code[code] for code in eligible_codes], xp, axis=0),
        len(buckets),
        rank_syncs,
        svd_fallbacks,
    )


class FamaMacBeth(BasePanelModel):
    """Fama-MacBeth two-pass regression estimator.

    The beta-series covariance remains estimator-specific. Stage B adds only
    parameter-based panel R-squared summaries; it deliberately does not route
    this estimator through residual-based OLS covariance or model-F machinery.
    """

    def __init__(
        self,
        cov_type: str = "newey-west",
        bandwidth: Optional[int] = None,
        alpha: float = 0.05,
        min_obs_per_period: int = 1,
        device: Union[str, Device] = Device.AUTO,
        n_jobs: Optional[int] = None,
    ):
        super().__init__(device=device, n_jobs=n_jobs)
        self.cov_type = str(cov_type).lower()
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.min_obs_per_period = min_obs_per_period
        if self.cov_type not in ("nonrobust", "newey-west"):
            raise ValueError("cov_type must be 'nonrobust' or 'newey-west'")
        self.fit_statistics_ = None

    def _reset_fit_state(self):
        """Invalidate all fit/inference outputs before a new fit attempt."""
        self._fitted = False
        self.fit_statistics_ = None
        for name in (
            "coef_",
            "bse_",
            "tvalues_",
            "pvalues_",
            "conf_int_",
            "betas_",
            "cov_params_",
            "nobs",
            "n_periods",
            "df_resid",
            "_params",
            "_bse",
            "_tvalues",
            "_zvalues",
            "_pvalues",
            "_conf_int",
            "_inference_result",
            "_backend_name",
            "_inference_backend_name",
            "_period_solver_mode",
            "_period_solver_batches",
            "_period_rank_syncs",
            "_period_svd_fallbacks",
            "_xp",
            "_fit_ref_",
            "_panel_index_info",
            "_design_info",
            "_feature_names",
            "_formula_has_intercept",
        ):
            self.__dict__.pop(name, None)

    def _validate_parameters(self):
        if self._cov_type not in ("nonrobust", "newey-west"):
            raise ValueError("cov_type must be 'nonrobust' or 'newey-west'")
        if self.bandwidth is not None:
            if (
                isinstance(self.bandwidth, bool)
                or not isinstance(self.bandwidth, (int, np.integer))
                or int(self.bandwidth) < 0
            ):
                raise ValueError("bandwidth must be a non-negative integer or None")
        if not np.isfinite(float(self.alpha)) or not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must be finite and strictly between 0 and 1")
        if (
            isinstance(self.min_obs_per_period, bool)
            or not isinstance(self.min_obs_per_period, (int, np.integer))
            or int(self.min_obs_per_period) < 1
        ):
            raise ValueError("min_obs_per_period must be a positive integer")

    def _prepare_backend_arrays(self, X, y, *, validate_finite=True):
        # AUTO preserves an already backend-native input. An explicit device is
        # authoritative instead: convert heterogeneous inputs to the requested
        # NumPy/CuPy/Torch backend rather than silently letting container type
        # override the public execution request.
        if self._device == Device.AUTO:
            backend_name = _detect_backend(X, self._get_compute_device())
            xp = _get_xp(backend_name)
            X_source = X
            y_source = y
            ref = None
            if backend_name == "torch":
                import torch

                if isinstance(X, torch.Tensor):
                    ref = X
                else:
                    dev = self._get_compute_device()
                    target = "cuda" if dev.value in ("torch", "cuda") else "cpu"
                    ref = torch.empty(0, dtype=torch.float64, device=target)
        else:
            backend = self._get_backend(backend="auto")
            backend_name = backend.name
            xp = backend.xp
            X_source = self._to_array(X, backend=backend_name)
            y_source = self._to_array(y, backend=backend_name)
            ref = X_source if backend_name in {"cupy", "torch"} else None

        X_arr = xp_asarray(X_source, dtype=xp.float64, xp=xp, ref_arr=ref)
        y_arr = xp_asarray(y_source, dtype=xp.float64, xp=xp, ref_arr=X_arr).ravel()
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2 or int(X_arr.shape[0]) == 0 or int(X_arr.shape[1]) == 0:
            raise ValueError("X must be a non-empty one- or two-dimensional array")
        if int(y_arr.shape[0]) != int(X_arr.shape[0]):
            raise ValueError("X and y must have the same number of observations")
        if validate_finite and (
            not _finite_all(X_arr, xp) or not _finite_all(y_arr, xp)
        ):
            raise ValueError("X and y must contain only finite values")
        return backend_name, xp, X_arr, y_arr

    def fit(
        self,
        X=None,
        y=None,
        time_ids=None,
        formula=None,
        data=None,
        entity_ids=None,
    ):
        # Refit is transactional: a failed new fit must never leave the prior
        # model/inference surface appearing valid for the attempted dataset.
        self._reset_fit_state()
        self._validate_parameters()
        # Preserve current public behavior: time_ids must be explicitly supplied;
        # FamaMacBeth does not infer it from formula tokens in Stage B.
        if time_ids is None:
            raise ValueError("time_ids is required for FamaMacBeth")

        (
            y_data,
            X_data,
            _fe_eids,
            _fe_tids,
            _fe_entity,
            _fe_time,
            aligned,
        ) = self._panel_prepare_formula_fit(
            formula,
            data,
            X,
            y,
            model_has_intercept=True,
            side_arrays={"time_ids": time_ids, "entity_ids": entity_ids},
        )
        time_ids = aligned["time_ids"]
        entity_ids = aligned["entity_ids"]
        if formula is not None:
            if not bool(self._formula_has_intercept):
                raise ValueError(
                    "FamaMacBeth always includes a period intercept; explicit "
                    "no-intercept formulas are not supported"
                )
            self._feature_names = [
                "Intercept",
                *list(self._feature_names or []),
            ]

        # Direct X/y calls have already passed BaseEstimator's public finite-input
        # guard.  Formula/Patsy paths construct new numerical arrays internally,
        # so retain the local finite check only for that path.
        backend_name, xp, X_arr, y_arr = self._prepare_backend_arrays(
            X_data,
            y_data,
            validate_finite=formula is not None,
        )
        n_orig = int(X_arr.shape[0])
        time_labels, time_codes = factorize_panel_metadata(
            time_ids,
            name="time_ids",
            expected_n=n_orig,
        )

        entity_codes = None
        if entity_ids is not None:
            entity_codes, _ = factorize_panel_labels(
                entity_ids,
                xp,
                ref_arr=X_arr,
                name="entity_ids",
                expected_n=n_orig,
            )
        self._panel_set_index_info(
            n_orig, entity_ids=entity_ids, time_ids=time_ids
        )

        counts = np.bincount(time_codes)
        intercept = xp_ones((n_orig, 1), xp.float64, xp, X_arr)
        X_design = (
            xp.cat([intercept, X_arr], dim=1)
            if xp.__name__ == "torch"
            else xp.concatenate([intercept, X_arr], axis=1)
        )
        k = int(X_design.shape[1])

        if backend_name in {"torch", "cupy"}:
            betas, solver_batches, rank_syncs, svd_fallbacks = (
                _gpu_certified_period_betas(
                    X_design,
                    y_arr,
                    time_codes,
                    counts,
                    time_labels,
                    min_obs_per_period=self.min_obs_per_period,
                    backend_name=backend_name,
                    xp=xp,
                )
            )
            if betas is None:
                raise ValueError("No time periods with enough observations")
            self._period_solver_mode = (
                "gram-certified"
                if int(svd_fallbacks) == 0
                else "gram-certified+svd-fallback"
            )
            self._period_solver_batches = int(solver_batches)
            self._period_rank_syncs = int(rank_syncs)
            self._period_svd_fallbacks = int(svd_fallbacks)
        else:
            betas_list = []
            for code, n_t in enumerate(counts):
                if int(n_t) < int(self.min_obs_per_period) or int(n_t) < k:
                    continue
                idx = _index_array(np.flatnonzero(time_codes == code), xp, X_design)
                X_t = X_design[idx]
                y_t = y_arr[idx]
                beta_t, rank_t = panel_lstsq(X_t, y_t, xp)
                if rank_t < k:
                    raise ValueError(
                        "FamaMacBeth requires full column rank in every retained period; "
                        f"retained time period {time_labels[code]!r} is rank deficient "
                        f"(rank={rank_t}, columns={k})"
                    )
                betas_list.append(beta_t)

            if not betas_list:
                raise ValueError("No time periods with enough observations")
            betas = _stack(betas_list, xp, axis=0)
            self._period_solver_mode = "serial"
            self._period_solver_batches = int(len(betas_list))
            self._period_rank_syncs = int(len(betas_list))
            self._period_svd_fallbacks = 0

        T = int(betas.shape[0])
        if T < 2:
            raise ValueError("FamaMacBeth requires at least 2 time periods after filtering")

        avg_beta = xp.mean(betas, axis=0)
        beta_centered = betas - avg_beta
        effective_bandwidth = None
        if self._cov_type == "nonrobust":
            covariance = (beta_centered.T @ beta_centered) / float(T - 1)
            cov_params = covariance / float(T)
        else:
            bandwidth = self.bandwidth
            if bandwidth is None:
                bandwidth = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
            bandwidth = max(0, min(int(bandwidth), T - 1))
            effective_bandwidth = bandwidth
            long_run = beta_centered.T @ beta_centered / float(T)
            for lag in range(1, bandwidth + 1):
                weight = 1.0 - lag / float(bandwidth + 1)
                gamma_lag = beta_centered[lag:].T @ beta_centered[:-lag] / float(T)
                long_run = long_run + weight * (gamma_lag + gamma_lag.T)
            cov_params = long_run / float(T)

        diagonal = xp.diag(cov_params)
        bse = xp.sqrt(xp.maximum(diagonal, xp.zeros_like(diagonal)))
        tvalues = avg_beta / bse
        df = T - 1

        dist_name = "normal" if self._cov_type == "newey-west" else "t"
        inference_device = (
            str(avg_beta.device)
            if backend_name == "torch" and hasattr(avg_beta, "device")
            else None
        )
        pvalues, critical = two_sided_reference_inference(
            xp.abs(tvalues),
            distribution=dist_name,
            alpha=self.alpha,
            backend=backend_name,
            xp=xp,
            df=None if dist_name == "normal" else df,
            device=inference_device,
        )
        conf_int = _stack(
            [avg_beta - critical * bse, avg_beta + critical * bse], xp, axis=1
        )

        self.coef_ = avg_beta
        self.bse_ = bse
        self.tvalues_ = tvalues
        self.pvalues_ = pvalues
        self.conf_int_ = conf_int
        self.betas_ = betas
        self.cov_params_ = cov_params
        self.nobs = n_orig
        self.n_periods = T
        self.df_resid = df
        self._backend_name = backend_name
        self._inference_backend_name = backend_name
        self._xp = xp
        self._fit_ref_ = xp_asarray([], dtype=xp.float64, xp=xp, ref_arr=X_arr)

        from statgpu.inference._results import ParameterInferenceResult

        reporting = _stack(
            [
                avg_beta,
                bse,
                tvalues,
                pvalues,
                conf_int[:, 0],
                conf_int[:, 1],
            ],
            xp,
            axis=0,
        )
        reporting_np = np.asarray(_to_numpy(reporting), dtype=np.float64)
        feature_names = getattr(self, "_feature_names", None)
        if feature_names is not None and len(feature_names) != int(avg_beta.shape[0]):
            feature_names = None
        inference = ParameterInferenceResult(
            method="fama_macbeth",
            feature_names=feature_names,
            metadata={
                "covariance_source": "period_coefficient_series",
                "n_periods": T,
                "effective_bandwidth": effective_bandwidth,
                "inference_backend": backend_name,
            },
            params=reporting_np[0],
            bse=reporting_np[1],
            statistic=reporting_np[2],
            statistic_name="z" if dist_name == "normal" else "t",
            pvalues=reporting_np[3],
            conf_int=np.stack([reporting_np[4], reporting_np[5]], axis=1),
            cov_type=self._cov_type,
            distribution="normal" if dist_name == "normal" else "t",
            df=None if dist_name == "normal" else float(df),
        )
        inference.apply_to(self)

        from statgpu.panel._diagnostics import _parameter_r2_components
        from statgpu.panel._results import PanelFitStatistics

        within, between, overall, degenerate = _parameter_r2_components(
            y_arr,
            X_design,
            avg_beta,
            xp=xp,
            entity_codes=entity_codes,
            has_constant=True,
        )
        unavailable = {
            "rsquared_adj": (
                "FamaMacBeth average-period adjusted R-squared is a distinct statistic "
                "and is not defined in Stage B"
            ),
            "model_f": (
                "FamaMacBeth beta-series joint inference is not a residual-OLS model F statistic"
            ),
        }
        if entity_codes is None:
            unavailable["within_between_r2"] = "entity_ids were not supplied"
        self.fit_statistics_ = PanelFitStatistics(
            rsquared_within=within,
            rsquared_between=between,
            rsquared_overall=overall,
            rsquared_adj=None,
            f_statistic=None,
            f_pvalue=None,
            f_df=None,
            metadata={
                "r2_definition": "parameter-based",
                "fit_space": "average FamaMacBeth coefficient on level panel",
                "degenerate_total_ss": degenerate,
                "unavailable": unavailable,
            },
        )

        self._fitted = True
        return self

    def predict(self, X):
        self._check_is_fitted()
        if self._design_info is None:
            X_data = X
        else:
            from statgpu.panel._formula import _formula_predict

            X_data = _formula_predict(
                X,
                self._design_info,
                self._formula_has_intercept,
                model_has_intercept=True,
            )
        xp = self._xp
        X_arr = xp_asarray(X_data, dtype=xp.float64, xp=xp, ref_arr=self._fit_ref_)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        if X_arr.ndim != 2 or int(X_arr.shape[1]) + 1 != int(self.coef_.shape[0]):
            raise ValueError("X has an incompatible feature count")
        intercept = xp_ones((int(X_arr.shape[0]), 1), xp.float64, xp, X_arr)
        X_design = (
            xp.cat([intercept, X_arr], dim=1)
            if xp.__name__ == "torch"
            else xp.concatenate([intercept, X_arr], axis=1)
        )
        return X_design @ self.coef_

    def summary(self):
        return self._panel_summary(
            model_type="FamaMacBeth",
            cov_type=self._cov_type,
        )

    def get_params(self, deep=True):
        """Return the shared exact-constructor parameter contract."""
        return super().get_params(deep)

    def set_params(self, **params):
        """Delegate parameter updates to the shared estimator contract."""
        return super().set_params(**params)
