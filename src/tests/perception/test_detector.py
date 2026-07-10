"""Detector seam: FakeDetector routing/scripting, GDINO prompt + lazy-import error."""
from __future__ import annotations

import builtins

import numpy as np
import pytest

from core.perception.detector import (
    Detection,
    FakeDetector,
    GDINO_REQUIRED_INSTALLS,
    GroundingDinoDetector,
    build_gdino_prompt,
)


def _tiles(n=4):
    return [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(n)]


# ------------------------------------------------------------------ Detection


def test_detection_center_and_foot():
    d = Detection(tile_id=0, bbox_xyxy=(10, 20, 30, 60), label="sofa", score=0.5)
    assert d.center_xy == (20.0, 40.0)
    assert d.foot_xy == (20.0, 60.0)


# ------------------------------------------------------------------ FakeDetector


def test_fake_detector_routes_to_tiles():
    dets = [
        Detection(0, (0, 0, 1, 1), "a", 0.5),
        Detection(2, (0, 0, 1, 1), "b", 0.5),
        Detection(0, (2, 2, 3, 3), "c", 0.5),
    ]
    fake = FakeDetector(dets)
    out = fake(_tiles(4))
    assert len(out) == 4
    assert [d.label for d in out[0]] == ["a", "c"]
    assert out[1] == []
    assert [d.label for d in out[2]] == ["b"]


def test_fake_detector_fixed_repeats():
    fake = FakeDetector([Detection(0, (0, 0, 1, 1), "x", 0.9)])
    a = fake(_tiles())
    b = fake(_tiles())
    assert [d.label for d in a[0]] == ["x"]
    assert [d.label for d in b[0]] == ["x"]  # same every call


def test_fake_detector_script_advances():
    script = [
        [Detection(0, (0, 0, 1, 1), "f1", 0.9)],
        [Detection(1, (0, 0, 1, 1), "f2", 0.9)],
    ]
    fake = FakeDetector(script=script)
    o0 = fake(_tiles())
    o1 = fake(_tiles())
    o2 = fake(_tiles())  # exhausted -> repeats last frame
    assert [d.label for d in o0[0]] == ["f1"]
    assert [d.label for d in o1[1]] == ["f2"]
    assert [d.label for d in o2[1]] == ["f2"]


def test_fake_detector_ignores_out_of_range_tile():
    fake = FakeDetector([Detection(9, (0, 0, 1, 1), "z", 0.5)])
    out = fake(_tiles(4))
    assert all(t == [] for t in out)


# ------------------------------------------------------------------ GDINO prompt


def test_build_gdino_prompt_dedupes_and_orders():
    prompt = build_gdino_prompt(["sofa", "window"], ["window", "potted plant", "sofa"])
    assert prompt == "sofa . window . potted plant ."


def test_build_gdino_prompt_empty():
    assert build_gdino_prompt([], []) == ""


def test_build_gdino_prompt_lowercases():
    assert build_gdino_prompt(["Sofa"], []) == "sofa ."


# ------------------------------------------------------------------ GDINO stub


def test_gdino_constructs_without_torch():
    # constructing must not import torch or raise
    det = GroundingDinoDetector(["sofa"], ["window"])
    assert "sofa" in det.prompt and "window" in det.prompt


def test_gdino_call_raises_clear_install_error(monkeypatch):
    """Calling without torch/groundingdino raises ImportError listing the pip installs."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("groundingdino"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    det = GroundingDinoDetector(["sofa"])
    with pytest.raises(ImportError) as exc:
        det([np.zeros((4, 4, 3), dtype=np.uint8)])
    msg = str(exc.value)
    for pkg in GDINO_REQUIRED_INSTALLS:
        assert pkg in msg
    assert "pip install" in msg
    assert "ubuntu_setup.md" in msg
