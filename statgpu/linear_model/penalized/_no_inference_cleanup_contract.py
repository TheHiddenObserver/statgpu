"""Exactly-once accelerator cleanup for estimation-only PGLM fits."""

from __future__ import annotations

from ._base import PenalizedGeneralizedLinearModel


_MISSING = object()


def _install_no_inference_cleanup_contract() -> None:
    """Deduplicate successful cleanup and cover pre-dispatch failures."""
    current = PenalizedGeneralizedLinearModel.fit
    if getattr(current, "_statgpu_no_inference_cleanup_contract", False):
        return

    def _fit_with_no_inference_cleanup_contract(
        self,
        X=None,
        y=None,
        sample_weight=None,
        formula=None,
        data=None,
    ):
        if (
            bool(getattr(self, "compute_inference", False))
            or not bool(getattr(self, "gpu_memory_cleanup", False))
        ):
            return current(
                self,
                X=X,
                y=y,
                sample_weight=sample_weight,
                formula=formula,
                data=data,
            )

        cleanup_calls = {"cupy": 0, "torch": 0}
        original_cleanup_cuda = self._cleanup_cuda_memory
        original_cleanup_torch = self._cleanup_torch_memory
        saved_instance_attrs = {
            "_cleanup_cuda_memory": self.__dict__.get(
                "_cleanup_cuda_memory", _MISSING
            ),
            "_cleanup_torch_memory": self.__dict__.get(
                "_cleanup_torch_memory", _MISSING
            ),
        }

        def _tracked_cleanup(backend_name, cleanup):
            cleanup_calls[backend_name] += 1
            if cleanup_calls[backend_name] > 1:
                return None
            return cleanup()

        def _cleanup_cuda_tracked():
            return _tracked_cleanup("cupy", original_cleanup_cuda)

        def _cleanup_torch_tracked():
            return _tracked_cleanup("torch", original_cleanup_torch)

        self._cleanup_cuda_memory = _cleanup_cuda_tracked
        self._cleanup_torch_memory = _cleanup_torch_tracked
        try:
            return current(
                self,
                X=X,
                y=y,
                sample_weight=sample_weight,
                formula=formula,
                data=data,
            )
        except Exception:
            # Conversion/device-alignment can fail before the inner fit
            # transaction reaches its own backend cleanup.  If the executed
            # backend is already known and no cleanup has run yet, supply the
            # missing best-effort release exactly once before re-raising.
            backend_name = str(
                getattr(self, "_selected_backend_name", "") or ""
            ).lower()
            if backend_name == "cupy" and cleanup_calls["cupy"] == 0:
                self._cleanup_cuda_memory()
            elif backend_name == "torch" and cleanup_calls["torch"] == 0:
                self._cleanup_torch_memory()
            raise
        finally:
            for name, previous in saved_instance_attrs.items():
                if previous is _MISSING:
                    self.__dict__.pop(name, None)
                else:
                    self.__dict__[name] = previous

    _fit_with_no_inference_cleanup_contract._statgpu_no_inference_cleanup_contract = True
    _fit_with_no_inference_cleanup_contract._statgpu_original = current
    PenalizedGeneralizedLinearModel.fit = _fit_with_no_inference_cleanup_contract


_install_no_inference_cleanup_contract()
