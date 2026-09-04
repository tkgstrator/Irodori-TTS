"""Which LoRA adapters stay on the device, and which get evicted to make room.

A full speaker set does not fit in VRAM alongside the base model: at r=16 an
adapter is ~53 MB, so a couple of hundred of them outweigh the checkpoint. The
runtime therefore keeps a bounded number loaded and brings the rest in on
demand, least-recently-used first out.

Loading and eviction are passed in as callbacks so this holds no reference to
peft or torch: the policy is what needs to be right, and it is testable on its
own.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Iterable


class AdapterResidency:
    """LRU bookkeeping over named adapters.

    `slots` of 0 means unbounded — every adapter is expected to be loaded
    already, and `ensure` does nothing.
    """

    def __init__(self, paths: dict[str, str], *, slots: int, resident: Iterable[str]) -> None:
        if slots < 0:
            raise ValueError(f"slots must not be negative: {slots}")
        self._paths = dict(paths)
        self._slots = int(slots)
        self._resident: OrderedDict[str, None] = OrderedDict((name, None) for name in resident)
        self._lock = threading.Lock()

    @property
    def slots(self) -> int:
        return self._slots

    @property
    def resident(self) -> list[str]:
        """Adapters on the device, least recently used first."""
        return list(self._resident)

    def path_for(self, name: str) -> str:
        try:
            return self._paths[name]
        except KeyError:
            raise KeyError(f"Unknown adapter: {name}") from None

    def ensure(
        self,
        name: str,
        *,
        load: Callable[[str, str], None],
        evict: Callable[[str], None],
    ) -> None:
        """Make `name` resident, evicting as needed. `load` gets (name, path)."""
        if not self._slots:
            return
        path = self.path_for(name)

        with self._lock:
            if name in self._resident:
                self._resident.move_to_end(name)
                return
            while len(self._resident) >= self._slots:
                evicted, _ = self._resident.popitem(last=False)
                evict(evicted)
            load(name, path)
            self._resident[name] = None
