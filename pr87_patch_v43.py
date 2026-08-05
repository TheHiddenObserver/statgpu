from pathlib import Path


def replace_once(path, old, new):
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected one match in {path}, found {count}")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path, marker, addition):
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if marker in text:
        return
    file_path.write_text(text.rstrip() + "\n\n" + addition.strip() + "\n", encoding="utf-8")


replace_once(
    "statgpu/solvers/_utils.py",
    '''        except (TypeError, ValueError, RuntimeError) as exc:\n            raise ValueError("sample_weight must be a real numeric array-like") from exc\n''',
    '''        except (TypeError, ValueError) as exc:\n            raise ValueError("sample_weight must be a real numeric array-like") from exc\n''',
)
replace_once(
    "statgpu/solvers/_utils.py",
    '''    except (TypeError, ValueError, RuntimeError) as exc:\n        raise ValueError("sample_weight must contain real finite values") from exc\n''',
    '''    except (TypeError, ValueError) as exc:\n        raise ValueError("sample_weight must contain real finite values") from exc\n''',
)

replace_once(
    "statgpu/solvers/_proximal_newton.py",
    '''"""Proximal Newton solver for smooth loss + non-smooth penalty.\n''',
    '''"""Newton solver with explicit non-smooth FISTA delegation.\n''',
)


tests = r'''
# PR87_REVIEW_FIX_V43
def test_solver_weight_validation_does_not_mask_runtime_failures(monkeypatch):
    import statgpu.solvers._utils as solver_utils

    class RuntimeFailingXP:
        @staticmethod
        def isfinite(values):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        solver_utils,
        "_native_sample_weight",
        lambda sample_weight: ("numpy", RuntimeFailingXP(), np.ones(2)),
    )
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        solver_utils._validate_sample_weight(np.ones(2), 2)


def test_solver_weight_validation_runtime_catches_are_narrow():
    from pathlib import Path

    source = Path("statgpu/solvers/_utils.py").read_text(encoding="utf-8")
    native_block = source.split("def _native_sample_weight", 1)[1].split(
        "def _validated_sample_weight", 1
    )[0]
    validated_block = source.split("def _validated_sample_weight", 1)[1].split(
        "def _validate_uniform_sample_weight", 1
    )[0]
    assert "except (TypeError, ValueError, RuntimeError)" not in native_block
    assert "except (TypeError, ValueError, RuntimeError)" not in validated_block
'''
append_once("dev/tests/test_maintenance_024_025.py", "# PR87_REVIEW_FIX_V43", tests)

replace_once(
    "CHANGELOG.md",
    "## Unreleased — maintenance hardening\n\n",
    "## Unreleased — maintenance hardening\n\n"
    "- Preserved backend RuntimeError failures (including CUDA OOM/device "
    "errors) during solver sample-weight validation instead of rewriting them "
    "as ordinary invalid-input ValueError exceptions.\n",
)
replace_once(
    "docs/en/changelog.md",
    "### Runtime safety\n\n",
    "### Runtime safety\n\n"
    "- Solver sample-weight validation now propagates backend RuntimeError "
    "failures such as CUDA OOM/device errors instead of masking them as "
    "invalid-input ValueError exceptions.\n",
)
replace_once(
    "docs/cn/changelog.md",
    "### 运行时安全\n\n",
    "### 运行时安全\n\n"
    "- solver sample-weight 校验现在会保留 CUDA OOM/device 等 backend "
    "RuntimeError，不再将其掩盖为普通输入 ValueError。\n",
)
