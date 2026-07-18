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
    GDINO_BACKOFF_BASE_S,
    GDINO_BACKOFF_CAP_S,
    GDINO_BACKOFF_DEGRADE_N,
    GDINO_MODEL_ID,
    GDINO_REQUIRED_INSTALLS,
    GroundingDinoDetector,
    _norm_cxcywh_to_tile_xyxy,
    build_gdino_prompt,
    refresh_prompt,
)


class _FakeClock:
    """Manually-advanced monotonic clock stand-in (mirrors mocks.mock_io.FakeClock)."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> float:
        self._t += float(dt)
        return self._t


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
        self.float32 = "float32"  # only used past a successful _ensure_model (backoff tests)


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


class _FakeModel:
    """Duck-typed stand-in for a loaded groundingdino model: records .to()/.eval()/.half()
    calls in order so tests can assert the device move happens before precision is applied."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def to(self, device):
        self.calls.append(f"to:{device}")
        return self

    def eval(self):
        self.calls.append("eval")
        return self

    def half(self):
        self.calls.append("half")
        return self


def test_ensure_model_moves_to_device_never_halves(tmp_path):
    # groundingdino-py's load_model(..., device="cuda") leaves the returned model on CPU
    # (issue #41); _ensure_model must move it explicitly. Weights are NEVER hard-.half()ed
    # (mixed float32 buffers inside GroundingDINO break under a halved model) — fp16 comes
    # from _forward_ctx's autocast instead, so resolved_half only arms that context.
    cfg = tmp_path / "gdino.cfg.py"
    cfg.write_text("# fake config\n")
    ckpt = tmp_path / "gdino.pth"
    ckpt.write_bytes(b"\x00")
    det = GroundingDinoDetector(
        config_path=str(cfg), checkpoint_path=str(ckpt), device="cuda", precision="auto",
    )
    fake_model = _FakeModel()
    model = det._ensure_model(_FakeTorch(cuda_available=True), lambda *a, **k: fake_model)
    assert model is fake_model
    assert "to:cuda" in fake_model.calls
    assert "half" not in fake_model.calls
    assert det._resolved_half is True  # arms the fp16 autocast forward context


class _FakeAutocastTorch(_FakeTorch):
    """Fake torch recording autocast constructions (returned ctx is a no-op)."""

    def __init__(self, cuda_available: bool) -> None:
        super().__init__(cuda_available)
        self.autocast_calls: list[dict] = []
        self.float16 = "float16"

    def autocast(self, device_type, dtype):
        self.autocast_calls.append({"device_type": device_type, "dtype": dtype})
        import contextlib

        return contextlib.nullcontext()


def test_forward_ctx_autocast_only_on_cuda_half(tmp_path):
    cfg = tmp_path / "gdino.cfg.py"
    cfg.write_text("# fake config\n")
    ckpt = tmp_path / "gdino.pth"
    ckpt.write_bytes(b"\x00")
    det = GroundingDinoDetector(
        config_path=str(cfg), checkpoint_path=str(ckpt), device="cuda", precision="auto",
    )
    fake_torch = _FakeAutocastTorch(cuda_available=True)
    det._ensure_model(fake_torch, lambda *a, **k: _FakeModel())
    with det._forward_ctx(fake_torch):
        pass
    assert fake_torch.autocast_calls == [{"device_type": "cuda", "dtype": "float16"}]

    det_cpu = GroundingDinoDetector(
        config_path=str(cfg), checkpoint_path=str(ckpt), device="cpu", precision="auto",
    )
    fake_torch_cpu = _FakeAutocastTorch(cuda_available=False)
    det_cpu._ensure_model(fake_torch_cpu, lambda *a, **k: _FakeModel())
    with det_cpu._forward_ctx(fake_torch_cpu):
        pass
    assert fake_torch_cpu.autocast_calls == []  # nullcontext on CPU/full precision


def test_ensure_model_no_half_on_cpu(tmp_path):
    cfg = tmp_path / "gdino.cfg.py"
    cfg.write_text("# fake config\n")
    ckpt = tmp_path / "gdino.pth"
    ckpt.write_bytes(b"\x00")
    det = GroundingDinoDetector(
        config_path=str(cfg), checkpoint_path=str(ckpt), device="cpu", precision="auto",
    )
    fake_model = _FakeModel()
    model = det._ensure_model(_FakeTorch(cuda_available=False), lambda *a, **k: fake_model)
    assert model is fake_model
    assert "to:cpu" in fake_model.calls
    assert "half" not in fake_model.calls


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


# ------------------------------------------------------------------ box clamping (verifier refutation)


def test_norm_cxcywh_to_tile_xyxy_centered_case_unchanged():
    # Sanity baseline for the clamp tests below: a fully-inside box is untouched.
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(0.5, 0.5, 0.5, 0.5, tile_w=640, tile_h=640)
    assert (x0, y0, x1, y1) == pytest.approx((160.0, 160.0, 480.0, 480.0))


def test_norm_cxcywh_to_tile_xyxy_edge_hugging_clamps_to_bounds():
    # cx=0, cy=0, w=0.5, h=0.5 on a 640x640 tile: unclamped affine gives (-160,-160,160,160)
    # — a box hugging the top-left corner whose far edge legitimately sits inside the tile.
    # Real GDINO detections at tile edges produce exactly this shape; downstream pixel
    # indexing must never see a negative coordinate.
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(0.0, 0.0, 0.5, 0.5, tile_w=640, tile_h=640)
    assert (x0, y0, x1, y1) == pytest.approx((0.0, 0.0, 160.0, 160.0))
    assert 0.0 <= x0 <= x1 <= 640.0
    assert 0.0 <= y0 <= y1 <= 640.0


def test_norm_cxcywh_to_tile_xyxy_fully_outside_degenerates_safely():
    # Entirely outside normalised [0, 1] space on both axes: every raw corner clamps to the
    # same tile edge, collapsing to a zero-area box AT the bound rather than an inverted
    # (x1 < x0) or out-of-bounds one.
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(-1.0, -1.0, 0.2, 0.2, tile_w=640, tile_h=640)
    assert (x0, y0, x1, y1) == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert x0 <= x1 and y0 <= y1  # never inverted

    # Fully outside past the high edge too.
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(2.0, 2.0, 0.2, 0.2, tile_w=640, tile_h=640)
    assert (x0, y0, x1, y1) == pytest.approx((640.0, 640.0, 640.0, 640.0))
    assert x0 <= x1 and y0 <= y1


# ------------------------------------------------------------------ prompt refresh (issue #34)


def test_fake_detector_has_empty_prompt_by_default():
    assert FakeDetector().prompt == ""


def test_refresh_prompt_updates_fake_detector_in_place():
    fake = FakeDetector()
    assert fake.prompt == ""
    new_prompt = refresh_prompt(fake, ["chair", "table"], ["sofa"])
    assert new_prompt == "chair . table . sofa ."
    assert fake.prompt == "chair . table . sofa ."  # mutated in place, same instance


def test_refresh_prompt_updates_gdino_detector_in_place():
    det = GroundingDinoDetector()  # boot-time construction: no question latched yet
    assert det.prompt == ""
    refresh_prompt(det, ["lamp"], [])
    assert det.prompt == "lamp ."


def test_refresh_prompt_none_detector_is_noop():
    assert refresh_prompt(None, ["chair"], []) is None


def test_refresh_prompt_detector_without_prompt_attr_is_noop():
    def plain_detector(tiles):
        return [[] for _ in tiles]

    assert refresh_prompt(plain_detector, ["chair"], []) is None


# ------------------------------------------------------------------ load backoff (issue #39)
# A permanently-failing model load used to re-run the full load_model cascade every
# perception tick (5 Hz), pegging adapter CPU and starving other subscribers. These tests
# drive GroundingDinoDetector with a fake clock + a stubbed ``_lazy_import`` returning a
# controllable (always-failing or always-succeeding) load_model_fn, so no torch/groundingdino
# install is needed.


def _backoff_detector(tmp_path, clock, load_model_fn):
    cfg = tmp_path / "gdino.cfg.py"
    cfg.write_text("# fake config\n")
    ckpt = tmp_path / "gdino.pth"
    ckpt.write_bytes(b"\x00")
    det = GroundingDinoDetector(
        ["sofa"], config_path=str(cfg), checkpoint_path=str(ckpt), device="cpu", clock=clock,
    )
    det._lazy_import = lambda: (_FakeTorch(cuda_available=False), load_model_fn, None)
    return det


class _FailingLoad:
    """Counts calls; always raises."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("boom: weights corrupt")


def test_backoff_empty_detections_during_cooldown_without_exception(tmp_path):
    clock = _FakeClock()
    load = _FailingLoad()
    det = _backoff_detector(tmp_path, clock, load)
    tiles = _tiles(4)

    out = det(tiles)  # first attempt fails

    assert out == [[], [], [], []]
    assert load.calls == 1
    assert det._consecutive_load_failures == 1


def test_backoff_schedule_no_reload_before_cooldown_then_reload_after(tmp_path):
    clock = _FakeClock()
    load = _FailingLoad()
    det = _backoff_detector(tmp_path, clock, load)
    tiles = _tiles(4)

    det(tiles)
    assert load.calls == 1  # first attempt

    clock.advance(0.5)  # < 1s backoff after first failure
    det(tiles)
    assert load.calls == 1  # still cooling down, no reload attempted

    clock.advance(0.5)  # now at the 1s boundary
    det(tiles)
    assert load.calls == 2  # cooldown elapsed -> retried (and failed again)

    clock.advance(1.9)  # < 2s backoff after second failure
    det(tiles)
    assert load.calls == 2

    clock.advance(0.1)  # now at the 2s boundary
    det(tiles)
    assert load.calls == 3


def test_backoff_success_resets_failure_state(tmp_path):
    clock = _FakeClock()
    load = _FailingLoad()
    det = _backoff_detector(tmp_path, clock, load)
    tiles = _tiles(4)

    det(tiles)
    clock.advance(GDINO_BACKOFF_BASE_S)
    det(tiles)
    assert det._consecutive_load_failures == 2

    fake_model = _FakeModel()
    clock.advance(2 * GDINO_BACKOFF_BASE_S)
    det._lazy_import = lambda: (_FakeTorch(cuda_available=False), lambda *a, **k: fake_model, None)
    det._call_batched = lambda torch, model, tiles, device, dtype: [[] for _ in tiles]
    det(tiles)

    assert det._consecutive_load_failures == 0
    assert det._next_retry_at == 0.0
    assert det._model is fake_model

    # No cooldown after a success: the very next call re-runs immediately.
    load2 = _FailingLoad()
    det._lazy_import = lambda: (_FakeTorch(cuda_available=False), load2, None)
    det._model = None  # force a fresh load attempt
    det(tiles)
    assert load2.calls == 1


def test_backoff_single_log_per_transition(tmp_path, caplog):
    clock = _FakeClock()
    load = _FailingLoad()
    det = _backoff_detector(tmp_path, clock, load)
    tiles = _tiles(4)

    with caplog.at_level("WARNING", logger="core.perception.detector"):
        for _ in range(GDINO_BACKOFF_DEGRADE_N):
            det(tiles)
            wait = det._next_retry_at - clock()
            if wait > 0:
                clock.advance(wait)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(warnings) == GDINO_BACKOFF_DEGRADE_N - 1
    assert len(errors) == 1
    assert "degraded" in errors[0].message
    assert load.calls == GDINO_BACKOFF_DEGRADE_N


def test_backoff_degrades_to_cap_after_n_failures_no_further_logging(tmp_path, caplog):
    clock = _FakeClock()
    load = _FailingLoad()
    det = _backoff_detector(tmp_path, clock, load)
    tiles = _tiles(4)

    with caplog.at_level("WARNING", logger="core.perception.detector"):
        for _ in range(GDINO_BACKOFF_DEGRADE_N):
            det(tiles)
            wait = det._next_retry_at - clock()
            if wait > 0:
                clock.advance(wait)
        # 5th failure just logged the single ERROR transition; delay is now capped.
        assert det._consecutive_load_failures == GDINO_BACKOFF_DEGRADE_N

        caplog.clear()
        # One more failure past the degrade point: still capped, no new log record.
        det(tiles)
        assert det._next_retry_at - clock() == pytest.approx(GDINO_BACKOFF_CAP_S)
        assert caplog.records == []
