# Node-wise alpha implementation notes

Baseline master: `8741857ff81c6fc5c90fbf32cd8aa72d63530881`
Branch: `fix/nodewise-alpha-inference-contract`

Implementation is in progress. This file records closure items discovered while wiring the reviewed plan into the current #138 runtime architecture.

- Keep runtime changes outside #134.
- Validate helper semantics on NumPy/CuPy/Torch before final review.
- Preserve #138 centered/weighted/fail-closed contracts.
- Final verdict requires a fresh exact-head review; plan review is not implementation approval.
