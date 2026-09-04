"""LRU residency policy for LoRA adapters.

The published speaker set does not fit in VRAM alongside the base model, so
the runtime keeps a bounded number of adapters loaded and brings the rest in
on demand. What has to be right is the policy — which adapter leaves when a
new one arrives — so these drive it with recording callbacks instead of a
model.
"""

from __future__ import annotations

import pytest

from irodori_tts.adapter_cache import AdapterResidency


class Recorder:
    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.evicted: list[str] = []

    def load(self, name: str, path: str) -> None:
        assert path.endswith(f"{name}.safetensors")
        self.loaded.append(name)

    def evict(self, name: str) -> None:
        self.evicted.append(name)


def residency(*, slots: int, names: list[str], resident: list[str] | None = None):
    paths = {name: f"/adapters/{name}.safetensors" for name in names}
    return AdapterResidency(paths, slots=slots, resident=resident or names[:1])


class TestAdapterResidency:
    def test_loads_on_demand_up_to_the_budget(self):
        rec = Recorder()
        r = residency(slots=3, names=["a", "b", "c", "d"])
        r.ensure("b", load=rec.load, evict=rec.evict)
        r.ensure("c", load=rec.load, evict=rec.evict)
        assert r.resident == ["a", "b", "c"]
        assert rec.evicted == []

    def test_evicts_the_least_recently_used(self):
        rec = Recorder()
        r = residency(slots=2, names=["a", "b", "c"])
        r.ensure("b", load=rec.load, evict=rec.evict)
        r.ensure("c", load=rec.load, evict=rec.evict)
        assert rec.evicted == ["a"]
        assert r.resident == ["b", "c"]

    def test_reuse_refreshes_recency(self):
        rec = Recorder()
        r = residency(slots=2, names=["a", "b", "c"])
        r.ensure("b", load=rec.load, evict=rec.evict)
        r.ensure("a", load=rec.load, evict=rec.evict)  # a is now the most recent
        r.ensure("c", load=rec.load, evict=rec.evict)
        assert rec.evicted == ["b"]
        assert r.resident == ["a", "c"]

    def test_a_resident_adapter_is_not_reloaded(self):
        rec = Recorder()
        r = residency(slots=2, names=["a", "b"])
        r.ensure("b", load=rec.load, evict=rec.evict)
        r.ensure("b", load=rec.load, evict=rec.evict)
        assert rec.loaded == ["b"]

    def test_zero_slots_never_loads_or_evicts(self):
        rec = Recorder()
        r = residency(slots=0, names=["a", "b"], resident=["a", "b"])
        r.ensure("b", load=rec.load, evict=rec.evict)
        assert (rec.loaded, rec.evicted) == ([], [])

    def test_one_slot_evicts_every_time(self):
        rec = Recorder()
        r = residency(slots=1, names=["a", "b", "c"])
        r.ensure("b", load=rec.load, evict=rec.evict)
        r.ensure("c", load=rec.load, evict=rec.evict)
        assert rec.evicted == ["a", "b"]
        assert r.resident == ["c"]

    def test_unknown_adapter_raises(self):
        rec = Recorder()
        r = residency(slots=2, names=["a", "b"])
        with pytest.raises(KeyError, match="Unknown adapter: zzz"):
            r.ensure("zzz", load=rec.load, evict=rec.evict)

    def test_negative_slots_rejected(self):
        with pytest.raises(ValueError, match="must not be negative"):
            AdapterResidency({"a": "/a"}, slots=-1, resident=["a"])

    def test_preloaded_set_beyond_the_budget_drains_to_it(self):
        # A runtime handed more resident adapters than slots (a shrunk budget
        # on reload) evicts down rather than growing past the cap.
        rec = Recorder()
        r = residency(slots=2, names=["a", "b", "c", "d"], resident=["a", "b", "c"])
        r.ensure("d", load=rec.load, evict=rec.evict)
        assert rec.evicted == ["a", "b"]
        assert r.resident == ["c", "d"]
