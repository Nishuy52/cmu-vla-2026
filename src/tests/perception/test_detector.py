"""Detector seam: FakeDetector routing/scripting, GDINO prompt + lazy-import error."""
from __future__ import annotations

import builtins

import numpy as np
import pytest

from core.perception.detector import (
    Detection,
    ENV_GDINO_CHECKPOINT_PATH,
    ENV_GDINO_CONFIG_PATH,
    ENV_GDINO_DEVICE,
    ENV_GDINO_MODEL_ID,
    ENV_GDINO_PRECISION,
    FakeDetector,
    GDINO_MODEL_ID,
    GDINO_REQUIRED_INSTALLS,
    GroundingDinoDetector,
    _norm_cxcywh_to_tile_xyxy,
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


# ------------------------------------------------------------------ GDINO real __call__ (Gate 4)
# The real forward-pass dispatch (_call_batched / _call_per_tile / _ensure_model) needs
# torch + groundingdino + baked weights, which per the Gate-4 environment rules must never
# be imported/loaded by a test run on this box (the sim owns the GPU). These tests instead
# cover every torch-free piece: the empty-input short circuits, the env/constructor
# precedence for every deploy-time knob, the device/precision resolution against a duck-
# typed stand-in for ``torch``, the actionable errors when weight paths are unset, and the
# pure box-space math that maps a model box back to tile pixels (the actual "map detections
# back through the tiling transform" logic). Real batched-vs-per-tile dispatch against
# real weights is listed in the post-recording checklist (nothing here loads torch).


def test_gdino_call_empty_tiles_short_circuits():
    det = GroundingDinoDetector(["sofa"])
    assert det([]) == []


def test_gdino_call_empty_prompt_short_circuits_without_torch():
    # No question nouns yet (freshly booted, no question latched) -> nothing to ground.
    # Must return per-tile empty lists WITHOUT ever needing torch/groundingdino installed.
    det = GroundingDinoDetector()  # no nouns -> prompt == ""
    assert det.prompt == ""
    out = det(_tiles(4))
    assert out == [[], [], [], []]


def test_gdino_model_id_precedence(monkeypatch):
    monkeypatch.delenv(ENV_GDINO_MODEL_ID, raising=False)
    assert GroundingDinoDetector().model_id == GDINO_MODEL_ID  # module default
    monkeypatch.setenv(ENV_GDINO_MODEL_ID, "env/model-id")
    assert GroundingDinoDetector().model_id == "env/model-id"  # env overrides default
    assert GroundingDinoDetector(model_id="ctor/model-id").model_id == "ctor/model-id"  # ctor wins


def test_gdino_precision_precedence_and_default(monkeypatch):
    monkeypatch.delenv(ENV_GDINO_PRECISION, raising=False)
    assert GroundingDinoDetector().precision == "auto"
    monkeypatch.setenv(ENV_GDINO_PRECISION, "fp32")
    assert GroundingDinoDetector().precision == "fp32"
    assert GroundingDinoDetector(precision="fp16").precision == "fp16"  # ctor wins over env


def test_gdino_device_env_honoured(monkeypatch):
    monkeypatch.delenv(ENV_GDINO_DEVICE, raising=False)
    assert GroundingDinoDetector()._device_pref is None
    monkeypatch.setenv(ENV_GDINO_DEVICE, "cpu")
    assert GroundingDinoDetector()._device_pref == "cpu"
    assert GroundingDinoDetector(device="cuda")._device_pref == "cuda"  # ctor wins over env


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    """Bare duck-typed stand-in for the one thing _resolve_device touches: cuda.is_available()."""

    def __init__(self, cuda_available: bool) -> None:
        self.cuda = _FakeCuda(cuda_available)


def test_resolve_device_prefers_explicit_pref():
    det = GroundingDinoDetector(device="cpu")
    assert det._resolve_device(_FakeTorch(cuda_available=True)) == "cpu"


def test_resolve_device_falls_back_to_cuda_availability():
    det = GroundingDinoDetector()
    assert det._resolve_device(_FakeTorch(cuda_available=True)) == "cuda"
    assert det._resolve_device(_FakeTorch(cuda_available=False)) == "cpu"


def test_resolve_half_fp16_and_fp32_explicit():
    assert GroundingDinoDetector(precision="fp16")._resolve_half("cpu") is True
    assert GroundingDinoDetector(precision="fp32")._resolve_half("cuda") is False


def test_resolve_half_auto_by_device():
    det = GroundingDinoDetector(precision="auto")
    assert det._resolve_half("cuda") is True  # half on CUDA: 8 GB dev box / 10-14 GB eval box
    assert det._resolve_half("cpu") is False  # full on CPU


def test_resolve_config_path_from_ctor_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_GDINO_CONFIG_PATH, raising=False)
    cfg = tmp_path / "gdino.cfg.py"
    cfg.write_text("# fake config\n")
    assert GroundingDinoDetector(config_path=str(cfg))._resolve_config_path() == str(cfg)
    monkeypatch.setenv(ENV_GDINO_CONFIG_PATH, str(cfg))
    assert GroundingDinoDetector()._resolve_config_path() == str(cfg)


def test_resolve_config_path_missing_raises_actionable_error(monkeypatch):
    monkeypatch.delenv(ENV_GDINO_CONFIG_PATH, raising=False)
    det = GroundingDinoDetector()
    # No baked config path and (in this venv) no installed groundingdino package to fall
    # back on -> a clear, actionable RuntimeError, not a bare AttributeError/ImportError.
    with pytest.raises(RuntimeError) as exc:
        det._resolve_config_path()
    msg = str(exc.value)
    assert "config_path" in msg
    assert ENV_GDINO_CONFIG_PATH in msg
    assert "ubuntu_setup.md" in msg


def test_resolve_checkpoint_path_from_ctor_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_GDINO_CHECKPOINT_PATH, raising=False)
    ckpt = tmp_path / "gdino.pth"
    ckpt.write_bytes(b"\x00")
    assert GroundingDinoDetector(checkpoint_path=str(ckpt))._resolve_checkpoint_path() == str(ckpt)
    monkeypatch.setenv(ENV_GDINO_CHECKPOINT_PATH, str(ckpt))
    assert GroundingDinoDetector()._resolve_checkpoint_path() == str(ckpt)


def test_resolve_checkpoint_path_missing_raises_actionable_error(monkeypatch):
    monkeypatch.delenv(ENV_GDINO_CHECKPOINT_PATH, raising=False)
    det = GroundingDinoDetector()
    with pytest.raises(RuntimeError) as exc:
        det._resolve_checkpoint_path()
    msg = str(exc.value)
    assert "checkpoint_path" in msg
    assert ENV_GDINO_CHECKPOINT_PATH in msg
    assert "baked into the Docker image" in msg  # offline-capable rule, never fetched at runtime


# ------------------------------------------------------------------ box-space math (tiling map-back)


def test_norm_cxcywh_to_tile_xyxy_center_box():
    # A box centred in a 480x640 tile spanning half its width/height.
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(0.5, 0.5, 0.5, 0.5, tile_w=480, tile_h=640)
    assert x0 == pytest.approx(120.0)
    assert x1 == pytest.approx(360.0)
    assert y0 == pytest.approx(160.0)
    assert y1 == pytest.approx(480.0)


def test_norm_cxcywh_to_tile_xyxy_full_frame():
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(0.5, 0.5, 1.0, 1.0, tile_w=100, tile_h=200)
    assert (x0, y0, x1, y1) == pytest.approx((0.0, 0.0, 100.0, 200.0))


def test_norm_cxcywh_to_tile_xyxy_scales_independently_per_axis():
    # A box that is a different fraction of width vs height lands correctly on each axis
    # independently — this is the actual math that stands in for "no inverse-resize needed".
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(0.25, 0.75, 0.1, 0.2, tile_w=200, tile_h=100)
    assert x0 == pytest.approx(40.0)
    assert x1 == pytest.approx(60.0)
    assert y0 == pytest.approx(65.0)
    assert y1 == pytest.approx(85.0)
