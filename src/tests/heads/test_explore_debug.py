"""issue #83/#84: explore_debug no-op guarantee + live-instance-index/keyframe dump."""
from __future__ import annotations

import json
import os

from core.heads import explore_debug
from core.heads.explore_debug import ENV_DEBUG_DIR, maybe_dump
from core.nav.occupancy import OccupancyGrid
from tests.heads._helpers import inst, scene


class _FakePolicy:
    min_frontier_score = 0.0


class _FakeHead:
    """Duck-types just enough of ExploreHead for maybe_dump."""

    def __init__(self, sc=None, perception=None):
        self._scene = sc
        self._policy = _FakePolicy()
        if perception is not None and sc is not None:
            sc._debug_perception = perception

    def _affinity(self):
        return lambda _xy: 0.0


def _grid() -> OccupancyGrid:
    return OccupancyGrid()


def test_noop_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_DEBUG_DIR, raising=False)
    head = _FakeHead(scene(inst(1, "chair")))
    maybe_dump(head, _grid(), (0.0, 0.0), 0.0, "sweep", None)
    # No file, no directory created, head untouched beyond what the caller set.
    assert list(tmp_path.iterdir()) == []


def test_dump_includes_live_instances_and_keyframes(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_DEBUG_DIR, str(tmp_path))

    class _Pipeline:
        _keyframe_idx = 7

    sc = scene(
        inst(1, "chair", n_obs=2, centroid=(1.0, 2.0, 0.0)),
        inst(2, "chair", n_obs=1, centroid=(3.0, 4.0, 0.0)),
        inst(3, "sofa", n_obs=5, centroid=(-1.0, 0.5, 0.0)),
    )
    head = _FakeHead(sc, perception=_Pipeline())
    maybe_dump(head, _grid(), (0.0, 0.0), 0.0, "frontier", (1.0, 1.0))

    files = list(tmp_path.glob("explore_debug_*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])

    li = record["live_instances"]
    assert li["total_instances"] == 3
    assert li["by_class"] == {"chair": 2, "sofa": 1}
    assert li["truncated"] is False
    assert len(li["instances"]) == 3
    ids = [i["id"] for i in li["instances"]]
    assert ids == sorted(ids)
    first = li["instances"][0]
    assert first["label"] == "chair"
    assert first["n_obs"] == 2
    assert first["position"] == [1.0, 2.0, 0.0]

    assert record["keyframes_processed"] == 7


def test_dump_handles_missing_scene_and_pipeline(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_DEBUG_DIR, str(tmp_path))
    head = _FakeHead(None)
    maybe_dump(head, _grid(), (0.0, 0.0), 0.0, "sweep", None)

    files = list(tmp_path.glob("explore_debug_*.jsonl"))
    record = json.loads(files[0].read_text().splitlines()[0])
    assert record["live_instances"] == {
        "total_instances": 0,
        "by_class": {},
        "instances": [],
        "truncated": False,
    }
    assert record["keyframes_processed"] is None


def test_instance_listing_truncated_at_cap(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_DEBUG_DIR, str(tmp_path))
    n = explore_debug.MAX_INSTANCES_LISTED + 5
    sc = scene(*[inst(i, "box", centroid=(float(i), 0.0, 0.0)) for i in range(n)])
    head = _FakeHead(sc)
    maybe_dump(head, _grid(), (0.0, 0.0), 0.0, "sweep", None)

    files = list(tmp_path.glob("explore_debug_*.jsonl"))
    record = json.loads(files[0].read_text().splitlines()[0])
    li = record["live_instances"]
    assert li["total_instances"] == n
    assert li["by_class"] == {"box": n}
    assert len(li["instances"]) == explore_debug.MAX_INSTANCES_LISTED
    assert li["truncated"] is True
