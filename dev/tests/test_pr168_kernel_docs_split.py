"""Protect the six-page kernel guide, complete API, and executable CPU lessons.

Analytic checks verify the actual documented code in fresh namespaces. Optional
CUDA examples are covered by the progressive syntax/dependency inventory only;
these CPU checks do not establish physical accelerator accuracy or placement.
"""

import ast
import inspect
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from doc_paths import markdown_anchors, model_doc, peer_link, reachable_pages
from doc_examples import example_code, parse_examples, run_example
from statgpu.nonparametric import kernel_methods as kernels

ROOT = Path(__file__).resolve().parents[2]
LESSONS = {
    "kernel-ridge": "kernel-ridge-cpu",
    "kernel-ridge-cv": "kernel-ridge-cv-cpu",
    "kernel-pca": "kernel-pca-cpu",
    "kernel-nystroem": "kernel-nystroem-cpu",
}
PAGES = ("models/nonparametric/kernel-methods", *(model_doc(page) for page in LESSONS),
         "reference/kernel-methods-api")
ENTRY_POINTS = ("README", "usage", "models/README", "guides/implemented-methods")
EXPORTS = (
    "KernelRidge", "KernelRidgeCV", "KernelPCA", "Nystroem", "pairwise_kernels",
    "rbf_kernel", "polynomial_kernel", "linear_kernel", "laplacian_kernel",
    "sigmoid_kernel", "cosine_kernel", "chi2_kernel",
)


def _read(language, page):
    return (ROOT / f"docs/{language}/{page}.md").read_text(encoding="utf-8")


def _run(language, page, name):
    return run_example(_read(language, page), name, f"docs/{language}/{page}.md")


def _rbf(X, Y, gamma=0.7):
    # Independent differences, not the implementation's norm/dot formula.
    return np.exp(-gamma * np.sum((X[:, None] - Y[None, :]) ** 2, axis=2))


def _anchors(text):
    return markdown_anchors(text)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_six_pages_separate_shared_concepts_method_lessons_and_complete_api(language):
    overview = _read(language, "models/nonparametric/kernel-methods")
    reference = _read(language, "reference/kernel-methods-api")
    assert set(parse_examples(overview)) == {"kernel-gram-cpu"}
    for page in LESSONS:
        assert f"]({page}.md" in overview
        tutorial = _read(language, model_doc(page))
        assert "](kernel-methods.md" in tutorial
        assert "](../../reference/kernel-methods-api.md" in tutorial
        assert f"](../{model_doc(page)}.md)" in reference
        assert "$$" in tutorial
    assert "](../../reference/kernel-methods-api.md)" in overview
    assert "](../models/nonparametric/kernel-methods.md)" in reference
    for name in EXPORTS[:4]:
        assert name + "(" not in overview, name
    assert "Gram" in overview and "gamma" in overview
    assert "1 / n_features" in overview and "1/16" in overview


@pytest.mark.parametrize("page,name", LESSONS.items())
def test_each_bilingual_method_example_has_the_same_self_contained_statements(page, name):
    en = _read("en", model_doc(page))
    cn = _read("cn", model_doc(page))
    for text in (en, cn):
        examples = parse_examples(text)
        assert name == next(iter(examples))
        assert not examples[name].requires
        assert examples[name].explicit_end
    # Translated comments may differ; every executable statement must match.
    assert ast.dump(ast.parse(example_code(en, name))) == ast.dump(ast.parse(example_code(cn, name)))


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", PAGES)
def test_each_page_has_a_valid_language_peer_and_public_audience(language, page):
    text = _read(language, page)
    other = "cn" if language == "en" else "en"
    assert f"]({peer_link(language, page)})" in text
    for phrase in ("contract closure", "exact-head", "maintained route", "review-clean",
                   "证据闭环", "维护中的路径", "PR #168"):
        assert phrase not in text


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page", ENTRY_POINTS)
def test_navigation_exposes_overview_every_method_and_complete_api(language, page):
    language_root = ROOT / "docs" / language
    reached = reachable_pages(language_root / (page + ".md"), language_root)
    for destination in ("kernel-methods", *LESSONS):
        expected = (language_root / (model_doc(destination) + ".md")).resolve()
        assert expected in reached, (page, destination)
    assert (language_root / "reference/kernel-methods-api.md").resolve() in reached


def test_root_readme_exposes_all_kernel_layers():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for page in ("kernel-methods", *LESSONS):
        assert f"](https://statgpu.github.io/en/{model_doc(page)})" in text
    assert "](https://statgpu.github.io/en/reference/kernel-methods-api)" in text


@pytest.mark.parametrize("language", ("en", "cn"))
def test_all_kernel_links_and_fragments_resolve_to_their_new_owners(language):
    paths = [ROOT / "README.md"] + list((ROOT / f"docs/{language}").rglob("*.md"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)]+)\)", text):
            link = urlsplit(target)
            if link.scheme or link.netloc:
                continue
            destination = (path.parent / unquote(link.path)).resolve() if link.path else path
            if not destination.stem.startswith("kernel-"):
                continue
            assert destination.is_file(), (path, target)
            if link.fragment:
                assert unquote(link.fragment) in _anchors(destination.read_text(encoding="utf-8")), (path, target)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_reference_covers_all_exports_parameters_methods_and_inherited_helpers(language):
    text = _read(language, "reference/kernel-methods-api")
    assert set(kernels.__all__) == set(EXPORTS)
    for name in EXPORTS:
        obj = getattr(kernels, name)
        assert name + "(" in text
        for parameter in inspect.signature(obj).parameters.values():
            assert re.search(r"\b" + parameter.name + r"\b", text), (name, parameter.name)
    for cls in (kernels.KernelRidge, kernels.KernelRidgeCV, kernels.KernelPCA, kernels.Nystroem):
        assert cls.__name__.lower() in _anchors(text)
        assert set(cls().get_params()) == set(inspect.signature(cls).parameters)
        for name, method in inspect.getmembers(cls, callable):
            if name.startswith("_"):
                continue
            assert re.search(r"\b" + name + r"\b", text), (cls.__name__, name)
    assert "](estimator-api.md" in text
    for key in ("alphas", "mean_mse", "mse_table", "mean_r2", "r2_table", "best_alpha", "best_score"):
        assert f"`{key}`" in text


@pytest.mark.parametrize("language", ("en", "cn"))
def test_gram_example_matches_analytic_similarity_and_query_orientation(language):
    scope = _run(language, "models/nonparametric/kernel-methods", "kernel-gram-cpu")
    expected = _rbf(scope["X"], scope["X"], np.log(2))
    assert scope["K"].shape == (3, 3)
    assert_allclose(scope["K"], expected, rtol=1e-13, atol=1e-15)
    assert_allclose(np.diag(scope["K"]), 1)
    assert np.linalg.eigvalsh(scope["K"]).min() > 0
    assert scope["K_query"].shape == (2, 3)
    assert_allclose(scope["K_query"], _rbf(scope["query"], scope["X"], np.log(2)), rtol=1e-13)


@pytest.mark.parametrize("language", ("en", "cn"))
@pytest.mark.parametrize("page,name", LESSONS.items())
def test_independent_lessons_keep_queries_out_of_fitted_training_state(language, page, name):
    scope = _run(language, model_doc(page), name)
    train, test = scope["train"], scope["test"]
    assert not np.intersect1d(train, test).size
    assert_array_equal(np.sort(np.r_[train, test]), np.arange(240))
    assert_allclose(scope["X_train"], scope["X"][train])
    assert_allclose(scope["X_test"], scope["X"][test])
    if page == "kernel-nystroem":
        model = scope["nystroem"]
        assert_allclose(model.components_, scope["X_train"][model.component_indices_])
    else:
        model = scope[{"kernel-ridge": "kr", "kernel-ridge-cv": "kr_cv", "kernel-pca": "kpca"}[page]]
        assert_allclose(model.X_fit_, scope["X_train"])


@pytest.mark.parametrize("language", ("en", "cn"))
def test_ridge_example_solves_the_unweighted_rkhs_system_and_predicts_queries(language):
    scope = _run(language, "models/nonparametric/kernel-ridge", "kernel-ridge-system-check")
    model, X, y = scope["kr"], scope["X_train"], scope["y_train"]
    K = _rbf(X, X)
    expected_dual = np.linalg.solve(K + model.alpha * np.eye(len(X)), y)
    assert_allclose(model.dual_coef_.ravel(), expected_dual, atol=1e-10)
    assert_allclose(scope["fixed_prediction"], _rbf(scope["X_test"], X) @ expected_dual, atol=1e-10)
    assert scope["fixed_prediction"].shape == (60,)
    assert model.score(scope["X_test"], scope["y_test"]) > 0.9
    assert_allclose(K @ (K @ expected_dual - y) + model.alpha * K @ expected_dual, 0, atol=1e-10)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_cv_example_selection_scores_and_final_refit_match_independent_fold_solves(language):
    scope = _run(language, "models/nonparametric/kernel-ridge-cv", "kernel-ridge-cv-cpu")
    model, X, y = scope["kr_cv"], scope["X_train"], scope["y_train"]
    K = _rbf(X, X)
    order = np.random.RandomState(42).permutation(len(X))
    folds = np.array_split(order, 5)
    mse = np.empty((4, 5, 1))
    r2 = np.empty_like(mse)
    for ai, alpha in enumerate(model.cv_results_["alphas"]):
        for fi, validation in enumerate(folds):
            training = np.concatenate([fold for j, fold in enumerate(folds) if j != fi])
            dual = np.linalg.solve(K[np.ix_(training, training)] + alpha * np.eye(len(training)), y[training])
            prediction = K[np.ix_(validation, training)] @ dual
            mse[ai, fi, 0] = np.mean((y[validation] - prediction) ** 2)
            r2[ai, fi, 0] = 1 - mse[ai, fi, 0] / np.var(y[validation])
    assert_allclose(model.cv_results_["mse_table"], mse, rtol=1e-8, atol=1e-11)
    assert_allclose(model.cv_results_["r2_table"], r2, rtol=1e-8, atol=1e-11)
    best = np.argmin(mse.mean(axis=(1, 2)))
    assert model.alpha_ == model.cv_results_["alphas"][best]
    assert model.best_score_ == pytest.approx(r2[best].mean())
    expected = _rbf(scope["X_test"], X) @ np.linalg.solve(K + model.alpha_ * np.eye(len(X)), y)
    assert_allclose(scope["prediction"], expected, atol=1e-10)
    assert scope["held_out_mse"] == pytest.approx(np.mean((scope["y_test"] - expected) ** 2))
    assert scope["held_out_r2"] > 0.9


@pytest.mark.parametrize("language", ("en", "cn"))
def test_external_cv_example_fits_scaling_inside_each_training_fold(language):
    scope = _run(language, "models/nonparametric/kernel-ridge-cv", "kernel-ridge-fold-local-cpu")
    X, y, folds, alphas = (scope[name] for name in ("X_train", "y_train", "folds", "alphas"))
    assert len(X) == 120 and len(scope["X_test"]) == 40
    expected = np.empty((len(alphas), len(folds)))
    for fi, validation in enumerate(folds):
        training = np.setdiff1d(np.arange(len(X)), validation)
        center, scale = X[training].mean(axis=0), X[training].std(axis=0)
        scale = np.where(scale > 0, scale, 1)
        fit = (X[training] - center) / scale
        query = (X[validation] - center) / scale
        K = _rbf(fit, fit)
        for ai, alpha in enumerate(alphas):
            prediction = _rbf(query, fit) @ np.linalg.solve(K + alpha * np.eye(len(fit)), y[training])
            expected[ai, fi] = np.mean((y[validation] - prediction) ** 2)
    assert_allclose(scope["losses"], expected, rtol=1e-9, atol=1e-11)
    alpha = alphas[np.argmin(expected.mean(axis=1))]
    assert scope["selected_alpha"] == alpha
    assert_allclose(scope["mean"], X.mean(axis=0))
    assert_allclose(scope["scale"], X.std(axis=0))
    fit = (X - X.mean(axis=0)) / X.std(axis=0)
    query = (scope["X_test"] - X.mean(axis=0)) / X.std(axis=0)
    prediction = _rbf(query, fit) @ np.linalg.solve(_rbf(fit, fit) + alpha * np.eye(len(X)), y)
    assert_allclose(scope["held_out_prediction"], prediction, atol=1e-10)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_pca_example_uses_training_centering_and_scaled_eigenvectors_for_queries(language):
    scope = _run(language, "models/nonparametric/kernel-pca", "kernel-pca-cpu")
    model, X = scope["kpca"], scope["X_train"]
    K = _rbf(X, X)
    centered = K - K.mean(axis=0) - K.mean(axis=1)[:, None] + K.mean()
    eigenvalues = np.linalg.eigvalsh(centered)[-3:][::-1]
    assert_allclose(model.lambdas_, eigenvalues, rtol=1e-10, atol=1e-11)
    vectors = model.alphas_ * np.sqrt(model.lambdas_)[None, :]
    assert_allclose(vectors.T @ vectors, np.eye(3), atol=1e-10)
    assert_allclose(centered @ vectors, vectors * model.lambdas_, atol=1e-10)
    assert_allclose(scope["train_coordinates"], centered @ model.alphas_, atol=1e-10)
    query = _rbf(scope["X_test"], X)
    query_centered = query - K.mean(axis=0) - query.mean(axis=1)[:, None] + K.mean()
    assert scope["test_coordinates"].shape == (60, 3)
    assert_allclose(scope["test_coordinates"], query_centered @ model.alphas_, atol=1e-10)


@pytest.mark.parametrize("language", ("en", "cn"))
def test_nystroem_example_preserves_landmarks_svd_orientation_and_approximation_error(language):
    scope = _run(language, "models/nonparametric/kernel-nystroem", "kernel-nystroem-cpu")
    model, X = scope["nystroem"], scope["X_train"]
    landmarks = X[model.component_indices_]
    assert model.components_.shape == (40, 2)
    assert np.unique(model.component_indices_).size == 40
    U, singular_values, Vt = np.linalg.svd(_rbf(landmarks, landmarks), full_matrices=False)
    normalization = (U / np.sqrt(np.maximum(singular_values, 1e-12))[None, :]) @ Vt
    assert_allclose(model.normalization_, normalization, rtol=1e-7, atol=1e-8)
    assert_allclose(model.eigenvalues_, np.maximum(singular_values, 1e-12), rtol=1e-7, atol=1e-12)
    assert_allclose(scope["train_features"], _rbf(X, landmarks) @ normalization, rtol=1e-7, atol=1e-8)
    assert_allclose(scope["test_features"], _rbf(scope["X_test"], landmarks) @ normalization, rtol=1e-7, atol=1e-8)
    exact = _rbf(scope["X_test"], scope["X_test"])
    approximation = scope["test_features"] @ scope["test_features"].T
    error = np.linalg.norm(exact - approximation) / np.linalg.norm(exact)
    assert scope["relative_kernel_error"] == pytest.approx(error, rel=1e-8)
    assert 0 <= error < 0.1
    features = scope["train_features"]
    weights = np.linalg.solve(features.T @ features + scope["downstream_alpha"] * np.eye(40),
                              features.T @ scope["y_train"])
    assert_allclose(scope["weights"], weights, atol=1e-10)
    assert_allclose(scope["prediction"], scope["test_features"] @ weights, atol=1e-10)
    assert scope["prediction"].shape == (60,)
    assert np.mean((scope["y_test"] - scope["prediction"]) ** 2) < 0.05


@pytest.mark.parametrize("language", ("en", "cn"))
def test_reference_callable_example_remains_independently_executable(language):
    text = _read(language, "reference/kernel-methods-api")
    examples = parse_examples(text)
    assert set(examples) == {"kernel-callable-cpu"}
    assert not examples["kernel-callable-cpu"].requires
    scope = _run(language, "reference/kernel-methods-api", "kernel-callable-cpu")
    assert_allclose(scope["implicit"], scope["X"] @ scope["X"].T)
    assert_allclose(scope["explicit"], scope["implicit"])


# Frozen URLs from the pre-split EN/CN overview, including implicit headings.
LEGACY_ANCHORS = {
    "en": {
        'a-training-and-held-out-cpu-workflow',
        'after-a-failed-refit',
        'are-kernelpca-components-directly-comparable-across-separate-fits',
        'backend-and-execution-boundaries',
        'build-a-smaller-kernel-feature-matrix',
        'built-in-kernels',
        'check-for-nonfinite-kernel-fits',
        'choose-a-tool-for-your-question',
        'compare-with-other-implementations',
        'complete-estimator-api',
        'complete-pairwise-function-api',
        'complexity-and-performance-notes',
        'computing-kernel-principal-components',
        'constructor-controls',
        'cupy',
        'does-deviceauto-silently-change-an-explicit-request',
        'evaluate-the-model',
        'faq',
        'fit-one-kernel-ridge-model',
        'fitted-attributes-and-outputs',
        'how-the-methods-work',
        'inference-semantics',
        'kernel-cpu-workflow',
        'kernel-cv-workflow',
        'kernel-methods',
        'kernel-pca',
        'kernel-ridge-cross-validation',
        'kernel-ridge-regression',
        'large-coordinate-offsets-with-rbf-kernels',
        'learn-and-reuse-nonlinear-coordinates',
        'limitations-and-failure-modes',
        'methods-and-shapes',
        'numerical-issues-and-practical-precautions',
        'nystroem-approximation',
        'optional-check-the-approximation-on-small-data',
        'optional-gpu-examples',
        'other-practical-precautions',
        'predict-and-interpret',
        'prepare-and-split-the-data',
        'reading-the-cross-validation-results',
        'references',
        'select-alpha-within-training-folds',
        'should-i-use-kernelridge-or-nystroem-plus-a-linear-model',
        'the-nystroem-feature-map',
        'torch-cuda',
        'validation-guidance',
        'why-can-a-gpu-kernel-method-be-slower-than-cpu',
    },
    "cn": {
        'cupy',
        'deviceauto-会覆盖显式请求吗',
        'faq',
        'kernel-cpu-workflow',
        'kernel-cv-workflow',
        'kernel-pca',
        'nystroem-的特征映射',
        'nystroem-近似',
        'rbf-坐标偏移过大时',
        'torch-cuda',
        '不同拟合的-kernelpca-成分是否可直接比较',
        '与其他实现核对结果',
        '为什么-gpu-核方法可能比-cpu-慢',
        '交叉验证结果的组织',
        '先按分析问题选择工具',
        '其他使用注意事项',
        '内置核',
        '准备并划分数据',
        '参考文献',
        '可选-gpu-示例',
        '可选检查小数据上的核近似',
        '后端与执行边界',
        '在训练折内选择-alpha',
        '复杂度与性能说明',
        '如何评价模型',
        '学习并复用非线性坐标',
        '完整估计器-api',
        '完整成对核函数-api',
        '应使用-kernelridge还是-nystroem-加线性模型',
        '拟合一个核岭模型',
        '拟合属性与输出',
        '推断语义',
        '数值问题与使用注意事项',
        '方法与形状',
        '构造参数',
        '构造较小的核特征矩阵',
        '核主成分的计算',
        '核岭交叉验证',
        '核岭回归',
        '核方法',
        '检查核拟合结果是否有限',
        '训练集与留出集的-cpu-流程',
        '进一步理解核方法',
        '重新拟合失败后',
        '限制与失败行为',
        '预测并解释结果',
        '验证方法',
    },
}


@pytest.mark.parametrize("language", ("en", "cn"))
def test_every_pre_split_overview_anchor_still_resolves(language):
    text = _read(language, "models/nonparametric/kernel-methods")
    assert len(LEGACY_ANCHORS[language]) == 47
    assert LEGACY_ANCHORS[language] <= _anchors(text)
    explicit = re.findall(r'<a id="([^\"]+)"></a>', text)
    assert len(explicit) == len(set(explicit)), "Duplicate explicit anchor IDs"
