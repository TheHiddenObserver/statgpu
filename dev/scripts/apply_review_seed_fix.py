"""Temporary seed compatibility patch for the repository review."""
from pathlib import Path

path = Path("statgpu/unsupervised/_utils.py")
text = path.read_text()
old = '''def draw_random_seed(random_state) -> int:
    """Draw an integer seed from int/None/RandomState/Generator inputs."""
    if random_state is None:
        return int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, np.iinfo(np.int32).max))
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(0, np.iinfo(np.int32).max))
    return int(random_state)
'''
new = '''def draw_random_seed(random_state) -> int:
    """Return a portable seed for NumPy, CuPy, and Torch generators.

    ``RandomState``-style generators accept unsigned 32-bit seeds. Drawing
    from that shared domain preserves fresh entropy for ``None`` while keeping
    the same seed usable by all supported backends.
    """
    max_seed = int(np.iinfo(np.uint32).max)
    if random_state is None:
        return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, max_seed, endpoint=True, dtype=np.uint32))
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(0, max_seed, dtype=np.uint32))
    seed = int(random_state)
    if seed < 0 or seed > max_seed:
        raise ValueError(f"random_state must be in [0, {max_seed}]")
    return seed
'''
if text.count(old) != 1:
    raise RuntimeError(f"draw_random_seed match count={text.count(old)}")
path.write_text(text.replace(old, new))

# Extend the staged regression file after batch1 creates it.
test_path = Path("dev/tests/test_repository_review_regressions.py")
