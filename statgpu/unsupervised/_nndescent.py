"""NNDescent: Approximate nearest neighbor search via iterative graph refinement.

Implements the NNDescent algorithm (Dong et al., 2011) for all 3 backends.
O(n log n) complexity instead of O(n²) for exact neighbors.

Algorithm:
1. Initialize: random k neighbors per point
2. Iterate: for each point, consider neighbors' neighbors as candidates
3. Keep k nearest candidates as new neighbors
4. Converge when neighbor graph stabilizes
"""
from __future__ import annotations

import numpy as np


def nndescent_numpy(X, k=15, max_iter=10, tol=0.001, seed=42):
    """NNDescent on CPU using numpy.

    Parameters
    ----------
    X : (n, d) array
        Input data.
    k : int
        Number of neighbors.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence threshold (fraction of neighbors changed).
    seed : int
        Random seed.

    Returns
    -------
    indices : (n, k) int64 array
        Neighbor indices.
    distances : (n, k) float64 array
        Neighbor distances.
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape

    # Initialize with random neighbors (avoid self)
    indices = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        candidates = list(range(n))
        candidates.remove(i)
        indices[i] = rng.choice(candidates, size=k, replace=False)

    # Compute initial distances
    distances = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        diff = X[i] - X[indices[i]]
        distances[i] = np.sum(diff * diff, axis=1)

    # Iterate
    for epoch in range(max_iter):
        # Collect candidates: neighbors + neighbors of neighbors
        all_candidates = set()
        for i in range(n):
            all_candidates.add(i)  # include self
            for j in indices[i]:
                all_candidates.add(j)
                for l in indices[j]:
                    all_candidates.add(l)

        # For each point, compute distances to candidates and keep k nearest
        changed = 0
        new_indices = np.zeros((n, k), dtype=np.int64)
        new_distances = np.zeros((n, k), dtype=np.float64)

        for i in range(n):
            candidates = list(all_candidates - {i})
            if len(candidates) > k:
                # Compute distances to candidates
                cand_X = X[candidates]
                diff = X[i] - cand_X
                dists = np.sum(diff * diff, axis=1)
                # Keep k nearest
                idx = np.argpartition(dists, k)[:k]
                sorted_idx = idx[np.argsort(dists[idx])]
                new_indices[i] = np.array(candidates)[sorted_idx]
                new_distances[i] = dists[sorted_idx]
            else:
                new_indices[i] = indices[i]
                new_distances[i] = distances[i]

            # Count changes
            changed += len(set(new_indices[i]) - set(indices[i]))

        # Check convergence
        change_ratio = changed / (n * k)
        if change_ratio < tol:
            break

        indices = new_indices
        distances = new_distances

    return indices, distances


def nndescent_torch(X, k=15, max_iter=10, tol=0.001, seed=42):
    """NNDescent on GPU using torch.

    Parameters
    ----------
    X : (n, d) tensor
        Input data on GPU.
    k : int
        Number of neighbors.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence threshold.
    seed : int
        Random seed.

    Returns
    -------
    indices : (n, k) int64 tensor
        Neighbor indices.
    distances : (n, k) float64 tensor
        Neighbor distances.
    """
    import torch

    n, d = X.shape
    device = X.device
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # Initialize with random neighbors (avoid self)
    indices = torch.zeros((n, k), dtype=torch.int64, device=device)
    for i in range(n):
        candidates = torch.arange(n, device=device)
        mask = candidates != i
        candidates = candidates[mask]
        perm = torch.randperm(len(candidates), generator=rng, device=device)[:k]
        indices[i] = candidates[perm]

    # Compute initial distances (use float64 for precision)
    X_neighbors = X[indices.view(-1)].view(n, k, d)
    diff = X.unsqueeze(1) - X_neighbors
    distances = torch.sum(diff * diff, dim=2).to(torch.float64)

    # Iterate
    node_ids = torch.arange(n, device=X.device).unsqueeze(1)  # (n, 1)
    for epoch in range(max_iter):
        # Collect candidates: neighbors + neighbors of neighbors
        nn_of_nn = indices[indices.view(-1)].view(n, k * k)
        candidates = torch.cat([indices, nn_of_nn], dim=1)  # (n, k + k²)

        # Exclude self-candidates (set distance to inf for self)
        is_self = (candidates == node_ids)  # (n, k + k²)

        # Compute distances to candidates
        X_cand = X[candidates.view(-1)].view(n, k + k * k, d)
        diff = X.unsqueeze(1) - X_cand
        dists = torch.sum(diff * diff, dim=2).to(torch.float64)  # (n, k + k²)
        dists[is_self] = float('inf')  # exclude self from top-k

        # Keep k nearest
        _, topk_idx = torch.topk(dists, k, largest=False, sorted=True)
        new_indices = candidates.gather(1, topk_idx)
        new_distances = dists.gather(1, topk_idx)

        # Check convergence
        changed = torch.sum(indices != new_indices).item()
        change_ratio = changed / (n * k)
        if change_ratio < tol:
            break

        indices = new_indices
        distances = new_distances

    return indices, distances


def nndescent_cupy(X, k=15, max_iter=10, tol=0.001, seed=42):
    """NNDescent on GPU using cupy.

    Parameters
    ----------
    X : (n, d) cupy array
        Input data on GPU.
    k : int
        Number of neighbors.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence threshold.
    seed : int
        Random seed.

    Returns
    -------
    indices : (n, k) int64 cupy array
        Neighbor indices.
    distances : (n, k) float64 cupy array
        Neighbor distances.
    """
    import cupy as cp

    n, d = X.shape
    dtype = X.dtype

    # Initialize with random neighbors (vectorized)
    rng = cp.random.RandomState(seed)
    indices = cp.zeros((n, k), dtype=cp.int64)
    for i in range(n):
        # Generate k unique random indices != i
        idx = rng.choice(n - 1, size=k, replace=False)
        idx = cp.where(idx >= i, idx + 1, idx)
        indices[i] = idx

    # Compute initial distances (vectorized)
    X_neighbors = X[indices.ravel()].reshape(n, k, d)
    diff = X[:, None, :] - X_neighbors
    distances = cp.sum(diff * diff, axis=2).astype(dtype)

    # Iterate
    node_ids = cp.arange(n, dtype=cp.int64).reshape(n, 1)  # (n, 1)
    for epoch in range(max_iter):
        # Collect candidates: neighbors + neighbors of neighbors (vectorized)
        nn_of_nn = indices[indices.ravel()].reshape(n, k * k)
        candidates = cp.concatenate([indices, nn_of_nn], axis=1)

        # Exclude self-candidates (set distance to inf for self)
        is_self = (candidates == node_ids)  # (n, k + k²)

        # Compute distances to candidates (vectorized)
        X_cand = X[candidates.ravel()].reshape(n, k + k * k, d)
        diff = X[:, None, :] - X_cand
        dists = cp.sum(diff * diff, axis=2).astype(dtype)
        dists[is_self] = cp.inf  # exclude self from top-k

        # Keep k nearest (vectorized)
        topk_idx = cp.argpartition(dists, k, axis=1)[:, :k]
        topk_dists = cp.take_along_axis(dists, topk_idx, axis=1)
        sort_idx = cp.argsort(topk_dists, axis=1)
        topk_idx = cp.take_along_axis(topk_idx, sort_idx, axis=1)

        new_indices = cp.take_along_axis(candidates, topk_idx, axis=1)
        new_distances = cp.take_along_axis(dists, topk_idx, axis=1)

        # Check convergence
        changed = int(cp.sum(indices != new_indices))
        change_ratio = changed / (n * k)
        if change_ratio < tol:
            break

        indices = new_indices
        distances = new_distances

    return indices, distances
