"""Lightweight memory system for workflow reuse (BioMedAgent pattern)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class AnalysisMemory:
    """Record of a past analysis workflow."""

    data_signature: str           # "n={n}_p={p}_task={task_type}"
    successful_methods: List[str]  # Methods that fitted without error
    failed_methods: List[str]      # Methods that failed + reason
    timestamp: str = ""            # ISO format timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)  # Extra info


class MemoryStore:
    """Persistent memory for workflow reuse.

    Stores successful analysis strategies so that future runs with
    similar data can benefit from past experience.
    """

    def __init__(self, path: str = "~/.statgpu/memory.json"):
        self.path = os.path.expanduser(path)
        self.memories: List[AnalysisMemory] = self._load()

    def _load(self) -> List[AnalysisMemory]:
        """Load memories from JSON file."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [AnalysisMemory(**m) for m in data]
            except (json.JSONDecodeError, TypeError, KeyError):
                return []
        return []

    def _save(self):
        """Save memories to JSON file."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([asdict(m) for m in self.memories], f, indent=2, ensure_ascii=False)

    def recall(self, data_signature: str) -> Optional[AnalysisMemory]:
        """Find the most recent memory matching the data signature."""
        for mem in reversed(self.memories):
            if mem.data_signature == data_signature:
                return mem
        return None

    def recall_similar(self, n_samples: int, n_features: int,
                       task_type: str) -> Optional[AnalysisMemory]:
        """Find the most similar past analysis by data shape."""
        target_sig = f"n={n_samples}_p={n_features}_task={task_type}"
        # Exact match first
        exact = self.recall(target_sig)
        if exact is not None:
            return exact
        # Fuzzy match: same task type, similar size
        for mem in reversed(self.memories):
            if f"task={task_type}" in mem.data_signature:
                return mem
        return None

    def save_memory(self, memory: AnalysisMemory):
        """Save a new memory entry."""
        if not memory.timestamp:
            memory.timestamp = datetime.now(timezone.utc).isoformat()
        self.memories.append(memory)
        # Keep only last 100 memories
        if len(self.memories) > 100:
            self.memories = self.memories[-100:]
        self._save()

    def record_analysis(self, n_samples: int, n_features: int,
                        task_type: str, successful: List[str],
                        failed: List[str], **metadata):
        """Convenience method to record an analysis result."""
        sig = f"n={n_samples}_p={n_features}_task={task_type}"
        memory = AnalysisMemory(
            data_signature=sig,
            successful_methods=successful,
            failed_methods=failed,
            metadata=metadata,
        )
        self.save_memory(memory)

    def clear(self):
        """Clear all memories."""
        self.memories = []
        self._save()

    def summary(self) -> Dict[str, Any]:
        """Return a summary of stored memories."""
        return {
            "total": len(self.memories),
            "path": self.path,
            "recent": [
                {"signature": m.data_signature, "methods": m.successful_methods}
                for m in self.memories[-5:]
            ],
        }
