"""Temporary reviewed patch script for PR #79. Removed after application."""
from pathlib import Path
from textwrap import dedent
import re


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"{path}: match count={text.count(old)} for {old[:50]!r}")
    p.write_text(text.replace(old, new))


replace_once(
    "statgpu/penalties/_adaptive_l1.py",
    "import numpy as np\nfrom statgpu.penalties._base import Penalty\n",
    "import numpy as np\nfrom statgpu.backends._array_ops import _xp\nfrom statgpu.penalties._base import Penalty\n",
)
replace_once(
    "statgpu/glm_core/_solver_utils.py",
    "        def _newton_eager(params, direction, params_old):\n"
    "            params_new = params - direction\n"
    "            diff_norm = torch.linalg.norm(params_new - params_old)\n"
    "            return params_new, diff_norm\n",
    "        def _newton_eager(params, direction, params_old):\n"
    "            import torch\n\n"
    "            params_new = params - direction\n"
    "            diff_norm = torch.linalg.norm(params_new - params_old)\n"
    "            return params_new, diff_norm\n",
)
replace_once(
    "statgpu/feature_selection/_knockoff_utils.py",
    '        use_cupy_native = str(backend_name).lower() == "cupy" and _is_cupy_array(Z)\n',
    '        use_cupy_native = str(backend_name).lower() == "cupy"\n',
)

p = Path("statgpu/survival/_cox.py")
text = p.read_text()
pattern = re.compile(
    r"        # Cumsum of outer products.*?"
    r'        hess \+= torch\.einsum\("g,gi,gj->ij", weights, E_X_at_uft, E_X_at_uft\)\n',
    re.S,
)
block = dedent(
    """
    # Sum weighted risk-set second moments without materializing an
    # O(n * p * p) tensor. Observation i contributes to every prefix
    # whose failure-time start is strictly after i.
    sc_at_start = torch.zeros(
        n_samples, dtype=torch.float64, device=beta.device
    )
    sc_at_start.index_add_(0, first_idx, sc)
    suffix_sc = torch.flip(
        torch.cumsum(torch.flip(sc_at_start, dims=[0]), dim=0),
        dims=[0],
    )
    prefix_weights = suffix_sc - sc_at_start
    weighted_prefix = X_exp.transpose(0, 1) @ (
        X * prefix_weights.unsqueeze(1)
    )

    hess = -torch.sum(sc) * total + weighted_prefix
    hess += torch.einsum(
        "g,gi,gj->ij", weights, E_X_at_uft, E_X_at_uft
    )
    """
)
block = "".join(("        " + line if line.strip() else line) for line in block.splitlines(True))
text, count = pattern.subn(block, text)
if count != 1:
    raise RuntimeError(f"Cox block count={count}")
p.write_text(text)

Path("statgpu/unsupervised/_nndescent.py").write_text(
    dedent(
        '''
        """NNDescent approximate nearest-neighbor search for NumPy, Torch, and CuPy."""
        from __future__ import annotations

        import numpy as np

        from statgpu.unsupervised._utils import draw_random_seed


        def _validate_inputs(X, k, max_iter, tol):
            if getattr(X, "ndim", None) != 2:
                raise ValueError("X must be a 2D array")
            n = int(X.shape[0])
            if n < 2:
                raise ValueError("X must contain at least two samples")
            if not isinstance(k, (int, np.integer)) or not 1 <= int(k) < n:
                raise ValueError("k must be an integer in [1, n_samples)")
            if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
                raise ValueError("max_iter must be a positive integer")
            if float(tol) < 0:
                raise ValueError("tol must be non-negative")
            return n, int(k), int(max_iter), float(tol)


        def nndescent_numpy(X, k=15, max_iter=10, tol=0.001, seed=42):
            n, k, max_iter, tol = _validate_inputs(X, k, max_iter, tol)
            d = int(X.shape[1])
            rng = np.random.RandomState(draw_random_seed(seed))
            indices = np.empty((n, k), dtype=np.int64)
            for i in range(n):
                choices = np.concatenate((np.arange(i), np.arange(i + 1, n)))
                indices[i] = rng.choice(choices, size=k, replace=False)
            neighbors = X[indices.reshape(-1)].reshape(n, k, d)
            distances = np.sum((X[:, None, :] - neighbors) ** 2, axis=2)
            for _ in range(max_iter):
                new_indices = np.empty((n, k), dtype=np.int64)
                new_distances = np.empty((n, k), dtype=np.float64)
                changed = 0
                for i in range(n):
                    candidates = set(int(v) for v in indices[i])
                    for neighbor in indices[i]:
                        candidates.update(int(v) for v in indices[int(neighbor)])
                    candidates.discard(i)
                    ids = np.fromiter(candidates, dtype=np.int64)
                    dists = np.sum((X[i] - X[ids]) ** 2, axis=1)
                    pos = np.argpartition(dists, k - 1)[:k]
                    pos = pos[np.argsort(dists[pos])]
                    new_indices[i] = ids[pos]
                    new_distances[i] = dists[pos]
                    changed += len(set(new_indices[i]) - set(indices[i]))
                indices, distances = new_indices, new_distances
                if changed / float(n * k) < tol:
                    break
            return indices, distances


        def nndescent_torch(X, k=15, max_iter=10, tol=0.001, seed=42):
            import torch
            n, k, max_iter, tol = _validate_inputs(X, k, max_iter, tol)
            d, device = int(X.shape[1]), X.device
            generator = torch.Generator(device=device)
            generator.manual_seed(draw_random_seed(seed))
            indices = torch.empty((n, k), dtype=torch.int64, device=device)
            all_ids = torch.arange(n, device=device)
            for i in range(n):
                choices = all_ids[all_ids != i]
                indices[i] = choices[torch.randperm(n - 1, generator=generator, device=device)[:k]]
            neighbors = X[indices.reshape(-1)].reshape(n, k, d)
            distances = torch.sum((X[:, None, :] - neighbors) ** 2, dim=2).to(torch.float64)
            node_ids = torch.arange(n, device=device).reshape(n, 1)
            for _ in range(max_iter):
                nn2 = indices[indices.reshape(-1)].reshape(n, k * k)
                candidates = torch.cat((indices, nn2), dim=1)
                order = torch.argsort(candidates, dim=1)
                sorted_ids = torch.gather(candidates, 1, order)
                dup_sorted = torch.zeros_like(sorted_ids, dtype=torch.bool)
                dup_sorted[:, 1:] = sorted_ids[:, 1:] == sorted_ids[:, :-1]
                duplicates = torch.zeros_like(dup_sorted)
                duplicates.scatter_(1, order, dup_sorted)
                invalid = (candidates == node_ids) | duplicates
                candidate_X = X[candidates.reshape(-1)].reshape(n, k + k * k, d)
                dists = torch.sum((X[:, None, :] - candidate_X) ** 2, dim=2).to(torch.float64)
                dists[invalid] = torch.inf
                new_distances, pos = torch.topk(dists, k, largest=False, sorted=True)
                new_indices = torch.gather(candidates, 1, pos)
                changed = int(torch.sum(indices != new_indices).item())
                indices, distances = new_indices, new_distances
                if changed / float(n * k) < tol:
                    break
            return indices, distances


        def nndescent_cupy(X, k=15, max_iter=10, tol=0.001, seed=42):
            import cupy as cp
            n, k, max_iter, tol = _validate_inputs(X, k, max_iter, tol)
            d = int(X.shape[1])
            rng = cp.random.RandomState(draw_random_seed(seed))
            indices = cp.empty((n, k), dtype=cp.int64)
            for i in range(n):
                choices = rng.choice(n - 1, size=k, replace=False)
                indices[i] = cp.where(choices >= i, choices + 1, choices)
            neighbors = X[indices.reshape(-1)].reshape(n, k, d)
            distances = cp.sum((X[:, None, :] - neighbors) ** 2, axis=2).astype(cp.float64)
            node_ids = cp.arange(n, dtype=cp.int64).reshape(n, 1)
            rows = cp.arange(n, dtype=cp.int64)[:, None]
            for _ in range(max_iter):
                nn2 = indices[indices.reshape(-1)].reshape(n, k * k)
                candidates = cp.concatenate((indices, nn2), axis=1)
                order = cp.argsort(candidates, axis=1)
                sorted_ids = cp.take_along_axis(candidates, order, axis=1)
                dup_sorted = cp.zeros_like(sorted_ids, dtype=cp.bool_)
                dup_sorted[:, 1:] = sorted_ids[:, 1:] == sorted_ids[:, :-1]
                duplicates = cp.zeros_like(dup_sorted)
                duplicates[rows, order] = dup_sorted
                invalid = (candidates == node_ids) | duplicates
                candidate_X = X[candidates.reshape(-1)].reshape(n, k + k * k, d)
                dists = cp.sum((X[:, None, :] - candidate_X) ** 2, axis=2).astype(cp.float64)
                dists[invalid] = cp.inf
                pos = cp.argpartition(dists, k - 1, axis=1)[:, :k]
                chosen = cp.take_along_axis(dists, pos, axis=1)
                pos = cp.take_along_axis(pos, cp.argsort(chosen, axis=1), axis=1)
                new_indices = cp.take_along_axis(candidates, pos, axis=1)
                new_distances = cp.take_along_axis(dists, pos, axis=1)
                changed = int(cp.sum(indices != new_indices))
                indices, distances = new_indices, new_distances
                if changed / float(n * k) < tol:
                    break
            return indices, distances
        '''
    )
)

p = Path("statgpu/unsupervised/_umap.py")
text = p.read_text()
text = text.replace(
    "    backend_random_normal,\n    check_2d_array,\n",
    "    backend_random_normal,\n    check_2d_array,\n    draw_random_seed,\n",
    1,
)
text = text.replace(
    "            seed = self.random_state if self.random_state is not None else 42\n",
    "            seed = int(self._fit_random_seed_)\n",
    1,
)
old_random = "backend_random_normal(backend, self.random_state, size="
if text.count(old_random) != 2:
    raise RuntimeError(f"UMAP random init count={text.count(old_random)}")
text = text.replace(old_random, "backend_random_normal(backend, self._fit_random_seed_, size=")
text = text.replace(
    "        self._validate_params(n_samples)\n\n        # Use float32",
    "        self._validate_params(n_samples)\n        self._fit_random_seed_ = draw_random_seed(self.random_state)\n\n        # Use float32",
    1,
)
text = text.replace(
    "        rng = np.random.RandomState(self.random_state)\n        rs = self.random_state if self.random_state is not None else 42\n",
    "        rs = int(self._fit_random_seed_)\n        rng = np.random.RandomState(rs)\n",
    1,
)
p.write_text(text)

Path("dev/tests/test_repository_review_regressions.py").write_text(
    dedent(
        '''
        import numpy as np
        import pytest

        from statgpu.penalties import AdaptiveL1Penalty
        from statgpu.unsupervised import UMAP
        from statgpu.unsupervised._nndescent import nndescent_numpy
        import statgpu.unsupervised._umap as umap_module


        def test_adaptive_l1_gradient_numpy():
            penalty = AdaptiveL1Penalty(
                alpha=2.0, weights=np.array([1.0, 3.0]), normalize=False
            )
            np.testing.assert_array_equal(
                penalty.gradient(np.array([-4.0, 5.0])), np.array([-2.0, 6.0])
            )


        def test_nndescent_numpy_unique_and_validated():
            X = np.random.default_rng(123).normal(size=(24, 4))
            indices, distances = nndescent_numpy(X, k=5, max_iter=3, seed=7)
            assert indices.shape == distances.shape == (24, 5)
            assert np.all(np.isfinite(distances))
            for i, row in enumerate(indices):
                assert i not in row
                assert len(np.unique(row)) == 5
            with pytest.raises(ValueError, match="k must"):
                nndescent_numpy(X, k=24)


        def test_umap_none_seed_draws_once_per_fit(monkeypatch):
            seeds = iter([101, 202])
            monkeypatch.setattr(umap_module, "draw_random_seed", lambda state: next(seeds))
            X = np.arange(60.0).reshape(20, 3)
            params = dict(
                n_neighbors=4,
                n_components=2,
                n_epochs=1,
                init="random",
                random_state=None,
                device="cpu",
            )
            first, second = UMAP(**params).fit(X), UMAP(**params).fit(X)
            assert (first._fit_random_seed_, second._fit_random_seed_) == (101, 202)
            assert not np.allclose(first.embedding_, second.embedding_)
        '''
    )
)
