"""Shared bounded-workspace helpers for Cox concordance calculations."""

from __future__ import annotations


MAX_CONCORDANCE_PAIR_ENTRIES = 2_000_000


def concordance_tile_shape(
    n_events: int,
    n_samples: int,
    *,
    max_pair_entries: int = MAX_CONCORDANCE_PAIR_ENTRIES,
) -> tuple[int, int]:
    """Return event/sample tile sizes whose product respects the hard limit."""
    limit = int(max_pair_entries)
    if limit < 1:
        raise ValueError("max_pair_entries must be a positive integer")
    event_count = max(int(n_events), 0)
    sample_count = max(int(n_samples), 0)
    sample_tile = max(1, min(sample_count, limit))
    event_tile = max(1, min(event_count, limit // sample_tile))
    return event_tile, sample_tile


__all__ = ["MAX_CONCORDANCE_PAIR_ENTRIES", "concordance_tile_shape"]
