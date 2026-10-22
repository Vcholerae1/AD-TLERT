"""LRU cache of solved fields shared by the 2.5D and 3D operators."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Generic, TypeVar

import numpy as np
import torch

Fields = TypeVar("Fields")
ModelKey = bytes | None


def step_batches(steps: Sequence[int], batch_steps: int) -> list[list[int]]:
    """Split ``steps`` into near-equal consecutive batches of at most ``batch_steps``."""

    if not steps:
        return []
    count = -(-len(steps) // batch_steps)
    return [batch.tolist() for batch in np.array_split(np.asarray(steps), count)]


def newest_batches_first(step_count: int, batch_steps: int) -> list[list[int]]:
    """Forward batches in reverse, so an adjoint pass starts with the factorization the solver
    still holds."""

    return step_batches(range(step_count), batch_steps)[::-1]


def factorization_name(keys: Sequence[ModelKey]) -> tuple[bytes, ...] | None:
    """Name of a batch's factorized operators, or ``None`` when any model is uncached."""

    named = tuple(key for key in keys if key is not None)
    return named if len(named) == len(keys) else None


class FieldCache(Generic[Fields]):
    """Fields of recently solved conductivity models, keyed by a hash of the model."""

    def __init__(self, max_entries: int) -> None:
        if int(max_entries) < 0:
            raise ValueError("normal_field_cache_max_entries must be non-negative")
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[bytes, Fields] = OrderedDict()
        self.hits = self.misses = 0

    def key(self, model: torch.Tensor) -> ModelKey:
        if self.max_entries < 1:
            return None
        host = np.ascontiguousarray(model.detach().cpu().numpy())
        digest = hashlib.blake2b(host.view(np.uint8), digest_size=16).digest()
        return digest + host.dtype.str.encode()

    def _lookup(self, key: ModelKey) -> Fields | None:
        if key is None:
            return None
        found = self._entries.get(key)
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
            self._entries.move_to_end(key)
        return found

    def _store(self, key: ModelKey, fields: Fields) -> None:
        if key is None:
            return
        self._entries[key] = fields
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def solve_series(
        self,
        models: torch.Tensor,
        batch_steps: int,
        solve: Callable[[torch.Tensor, tuple[bytes, ...] | None], Sequence[Fields]],
    ) -> tuple[list[Fields], list[ModelKey]]:
        """Fields and keys of every row of ``models``; misses are solved ``batch_steps`` at a time.

        ``solve(batch_models, factorization_name)`` returns the fields of one batch.
        """

        keys = [self.key(model) for model in models]
        fields = {
            step: found
            for step, key in enumerate(keys)
            if (found := self._lookup(key)) is not None
        }
        missing = [step for step in range(len(keys)) if step not in fields]
        for batch in step_batches(missing, batch_steps):
            name = factorization_name([keys[step] for step in batch])
            for step, solved in zip(batch, solve(models[batch], name), strict=True):
                fields[step] = solved
                self._store(keys[step], solved)
        return [fields[step] for step in range(len(keys))], keys

    def clear(self) -> None:
        self._entries.clear()

    def info(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
        }
