from pathlib import Path


def patch(path, old, new):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"anchor missing in {path}: {old[:80]!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


patch(
    "docs/en/panel/fama-macbeth.md",
    "On GPU, the Gram certificate is evaluated far from that numerical boundary and only selects whether the coefficient solve may use the fast path.",
    "On every maintained backend, the Gram certificate is evaluated far from that numerical boundary and only selects whether the coefficient solve may use the fast path.",
)
patch(
    "docs/en/panel/fama-macbeth.md",
    "The accepted PR126 P100 evidence is anchored to numerical source `8c60db00...` and is complemented by the exact-source Stage-C matrix and HAC-chronology runners.",
    "The historical PR126 P100 evidence remains anchored to numerical source `8c60db00...` and is complemented by the exact-source Stage-C matrix and HAC-chronology runners; it is not current-head acceptance after the later numerical-path fixes.",
)
patch(
    "docs/cn/panel/fama-macbeth.md",
    "GPU 上的 Gram certificate 被刻意放在远离 numerical rank boundary 的区域，它只决定 coefficient solve 是否可以使用 fast path；",
    "在所有 maintained backend 上，Gram certificate 都被刻意放在远离 numerical rank boundary 的区域，它只决定 coefficient solve 是否可以使用 fast path；",
)
patch(
    "docs/cn/panel/fama-macbeth.md",
    "PR126 accepted P100 evidence 统一锚定 numerical source `8c60db00...`，并由同一 source 的 Stage-C matrix 和 HAC-chronology runner 补齐 broad physical acceptance。",
    "PR126 的历史 P100 evidence 仍锚定 numerical source `8c60db00...`，并由同一 source 的 Stage-C matrix 和 HAC-chronology runner 补齐当时的 broad physical acceptance；后续 numerical-path 修复后，它不再代表 current-head acceptance。",
)
print("PR126 final Fama-MacBeth wording cleanup applied")
