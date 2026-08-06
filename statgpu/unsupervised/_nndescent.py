
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
