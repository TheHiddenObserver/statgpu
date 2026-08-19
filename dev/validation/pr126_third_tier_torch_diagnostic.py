import numpy as np
import torch

import statgpu.panel._covariance as covmod


def fixture():
    amplitude = 2.0 ** 660
    middle = 2.0 ** 600
    tiny = 2.0 ** 350
    scores_np = np.asarray(
        [
            -amplitude, middle, tiny, amplitude, -middle, -tiny,
            -amplitude, -middle, amplitude, middle,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        dtype=np.float64,
    )
    cluster1 = np.asarray(
        [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9],
        dtype=np.int64,
    )
    cluster2 = np.asarray(
        [0, 1, 1, 0, 1, 1, 2, 3, 2, 3, 4, 5, 6, 7, 8, 9],
        dtype=np.int64,
    )
    X = torch.full((16, 1), 0.5, dtype=torch.float64)
    resid = torch.as_tensor(8.0 * scores_np, dtype=torch.float64)
    expected = -4.0 * amplitude * tiny
    return X, resid, cluster1, cluster2, expected


def scalar(x):
    return float(np.asarray(x.detach().cpu().numpy()).reshape(-1)[0])


def run_public(label, monkeypatch_guard=False):
    X, resid, c1, c2, expected = fixture()
    original = covmod._validate_covariance_finite_inputs
    if monkeypatch_guard:
        covmod._validate_covariance_finite_inputs = lambda X, resid, xp: None
    try:
        values = []
        for _ in range(5):
            actual = covmod.two_way_clustered_covariance(
                X, resid, c1, c2, xp=torch
            )
            values.append(scalar(actual))
        print(label, "threads", torch.get_num_threads(), "expected", expected, "values", values)
    finally:
        covmod._validate_covariance_finite_inputs = original


def components_after_guard(call_guard):
    X, resid, cluster1, cluster2, expected = fixture()
    if call_guard:
        covmod._validate_covariance_finite_inputs(X, resid, torch)
    n = int(X.shape[0])
    labels1, c1 = covmod._factorize_1d_labels(cluster1, nobs=n, name="cluster1")
    labels2, c2 = covmod._factorize_1d_labels(cluster2, nobs=n, name="cluster2")
    c12 = covmod._paired_codes(c1, c2)
    n12 = int(np.max(c12)) + 1
    influence, projection_scale, design_scale, *_ = covmod._influence_rows(X, resid, torch)
    sets = []
    corrections = []
    for codes, ng in ((c1, len(labels1)), (c2, len(labels2)), (c12, n12)):
        comps, correction = covmod._cluster_grouped_scores(
            influence, codes, n_groups=int(ng), nobs=n,
            group_debias=False, xp=torch, return_components=True
        )
        sets.append(comps)
        corrections.append(correction)
    need = covmod._component_row_reduction_needs_expansion(tuple(sets), torch)
    refined = covmod._retier_component_sets_for_safe_gram(tuple(sets), torch) if need else tuple(sets)
    print("components", "guard", call_guard, "need_retier", need,
          "lens", [len(v) for v in sets], "refined", [len(v) for v in refined],
          "projection_scale", projection_scale.detach().cpu().numpy().tolist(),
          "design_scale", design_scale)
    for si, comps in enumerate(refined):
        for ci, comp in enumerate(comps):
            arr = comp.detach().cpu().numpy().reshape(-1)
            nz = arr[arr != 0.0]
            print("set", si, "component", ci, "nonzero", nz.tolist())

    all_components = tuple(x for comps in refined for x in comps)
    max_rows = max(int(comps[0].shape[0]) for comps in refined)
    product_count = sum(int(comps[0].shape[0]) * (len(comps) ** 2) for comps in refined)
    multiplier = 2.0 * float(max(1, product_count)) / float(max(1, max_rows))
    work, scale = covmod._common_gram_working_values(
        list(all_components), torch, max_multiplier=multiplier
    )
    print("common scale", scale.detach().cpu().numpy().tolist(),
          "product_count", product_count, "multiplier", multiplier)
    n1, n2 = len(refined[0]), len(refined[1])
    work_sets = (work[:n1], work[n1:n1+n2], work[n1+n2:])
    terms = []
    for comps, sign in zip(work_sets, (1.0, 1.0, -1.0)):
        for i, left in enumerate(comps):
            terms.append(covmod._symmetrize(left.T @ left) * sign)
            for right in comps[:i]:
                cross = left.T @ right
                terms.append((cross + cross.T) * sign)
    vals = [scalar(t) for t in terms]
    print("terms", vals)
    summed = covmod._stable_matrix_expansion_sum(terms, torch)
    print("cov_work", scalar(summed), "restored", scalar(covmod._restore_influence_covariance(
        summed, scale, projection_scale, design_scale, torch
    )), "expected", expected)


print("torch", torch.__version__, "threads", torch.get_num_threads())
run_public("guard-on", monkeypatch_guard=False)
run_public("guard-off", monkeypatch_guard=True)
components_after_guard(False)
components_after_guard(True)
for threads in (1, 2, 4):
    torch.set_num_threads(threads)
    run_public(f"guard-on-threads-{threads}", monkeypatch_guard=False)
