"""Adapter seam-wiring guard (H6/H8/H14) — Windows-runnable via source inspection + import.

The adapter node imports rclpy at module top, so on the Windows dev box it cannot be
imported (see test_importable.py). These tests therefore verify the seam wiring two ways:

* Source inspection (always runs): the load-bearing wiring lines from H6/H8/H14 are present
  and structured as required (perception pipeline seam, VLA_DETECTOR factory, ladder parse,
  checkpoint seams, budget_frac/remaining_s late-bound closures, the .npy/provider boot
  assert, the second-question loud-log policy, the SUBMISSION-BLOCKER shout).
* Real import (Ubuntu only, skipped on Windows): construct nothing (needs a ROS graph) but
  confirm the new module-level symbols exist once rclpy is importable.

Behavioural verification of the seams themselves lives at the core level
(tests/heads/test_factory_timeout.py, tests/fsm/test_ledger_admission.py) where the code is
ROS-free and fully exercisable; this file guards that the adapter actually wires those seams.
"""
from __future__ import annotations

import pathlib

import pytest

_ADAPTER = pathlib.Path(__file__).resolve().parents[2] / "ros_adapter" / "adapter_node.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _ADAPTER.read_text(encoding="utf-8")


# --------------------------------------------------------------------- H6 perception seam


def test_perception_pipeline_seam_present(src: str):
    assert "from core.perception.tracker import PerceptionPipeline" in src
    assert "def make_detector(" in src
    assert 'VLA_DETECTOR' in src or "ENV_DETECTOR" in src
    # The controller must see the pipeline's LIVE index when perception is on.
    assert "self._scene_index = self._perception.index" in src
    # New-pano-gated feed following the runner/single.py scripted pattern.
    assert "def _maybe_process_perception(" in src
    assert "self._perception.process(pano, scan)" in src


def test_detector_factory_defaults_to_stub(src: str):
    # VLA_DETECTOR=none -> None (stub), keeping today's empty-index behaviour + the shout.
    assert 'DETECTOR_NONE = "none"' in src
    assert 'DETECTOR_GDINO = "grounding_dino"' in src
    assert "GroundingDinoDetector()" in src
    # SUBMISSION-BLOCKER shout retained and gated on the stub path.
    assert "SUBMISSION-BLOCKER: scene index is the empty stub" in src
    assert "instances_tracked > 0 on a live scene" in src


# --------------------------------------------------------------------- H8 LLM / checkpoints


def test_ladder_and_config_wired(src: str):
    assert "from core.llm.config import build_chat_fns, load_config" in src
    assert "from core.parsing import ladder as parse_ladder" in src
    assert "self._chat_fns = build_chat_fns(self._llm_config)" in src
    # parse routed through the ladder with the controller's clock + ledger.
    assert "parse_ladder.parse(question, self._chat_fns, clock, ledger)" in src


def test_checkpoint_seams_wired(src: str):
    for builder in (
        "build_verifier",
        "build_anchor_confirm",
        "build_miss_recovery",
        "build_frontier_select",
    ):
        assert builder in src, f"missing checkpoint builder wiring: {builder}"
    # Seams passed to build_callables by their rich-seam kwarg names.
    assert "verifier=seams.get(\"verifier\")" in src
    assert "anchor_confirmer=seams.get(\"anchor_confirmer\")" in src
    assert "miss_recoverer=seams.get(\"miss_recoverer\")" in src
    assert "frontier_selector=seams.get(\"frontier_selector\")" in src


def test_budget_frac_and_remaining_late_bound(src: str):
    # budget_frac / remaining_s are late-bound closures over the controller's BudgetState
    # (created at intake) and passed to build_callables on BOTH the configured + offline paths.
    assert "def _budget_frac(" in src
    assert "def _remaining_s(" in src
    assert "budget_frac=_budget_frac" in src
    assert "remaining_s=_remaining_s" in src
    # Even offline (no LLM) budget_frac must be wired so the H4c gate is live.
    assert "if not self._llm_configured:" in src


def test_local_tier_descoped(src: str):
    # The dark-network local tier is explicitly descoped (comment), api->api2->regex shipped.
    low = src.lower()
    assert "descope" in low
    assert "local tier" in low


# --------------------------------------------------------------------- H14 hygiene


def test_encoder_provider_boot_assert(src: str):
    assert "def _assert_encoder_provider_consistency(" in src
    assert "self._assert_encoder_provider_consistency()" in src
    # Must key off a real network provider kind, not just "any config".
    assert '{"openai", "anthropic"}' in src or "network_kinds" in src
    assert ".npy" in src


def test_second_question_loud_log_policy(src: str):
    # Different second question: keep the first, log LOUDLY (error), documented deliberate policy.
    assert "SECOND QUESTION DROPPED" in src
    assert "keeping the first" in src.lower() or "keeping the first" in src


# --------------------------------------------------------------------- coordinator seam args


def test_cp2_tile_dims_wired(src: str):
    # OR-F8: CP2 bbox-bounds rejection needs the real tile pixel dims at build time.
    assert "def _tile_pixel_dims(" in src
    assert 'build_miss_recovery(' in src
    assert '"tile_w": tile_w, "tile_h": tile_h' in src


def test_cp4_rich_verifier_carries_runner_up(src: str):
    # The rich verifier (build_verifier) is injected as `verifier`, and its contract carries
    # the runner-up (the ObjectRefHead calls it with runner_up=...), so the CP4 prompt gets a
    # real runner-up rather than "(none)". We inject the rich seam, not the bool as_llm_verify.
    assert "build_verifier(text_chat, ledger_proxy, clock)" in src
    assert 'verifier=seams.get("verifier")' in src


def test_cp3_anchor_seam_built(src: str):
    # The CP3 seam is built and accepts the optional anchor_noun the caller wires through,
    # plus the real vision encode_fn (Gate 4 item 2) so a configured network provider gets
    # real JPEG bytes rather than the checkpoint's raw-.npy default.
    assert "build_anchor_confirm(" in src
    assert '"encode_fn": self._vision_encode_fn' in src
    assert 'anchor_confirmer=seams.get("anchor_confirmer")' in src


def test_cp5_frontier_seam_gets_vision_encoder(src: str):
    # CP5 (frontier_select) is also a vision checkpoint (Gate 4 item 2) — it must get the
    # same real-JPEG-when-available encode_fn as CP2/CP3, not just the text ladder.
    assert "build_frontier_select(" in src
    assert src.count('"encode_fn": self._vision_encode_fn') >= 3  # CP2, CP3, CP5


def test_vision_encode_fn_resolved_at_boot(src: str):
    # Gate 4 item 2: a real JPEG encode_fn is resolved once at boot (Pillow-lazy,
    # gracefully degrading to None when Pillow is absent — see core.perception.vision_encode).
    assert "from core.perception.vision_encode import resolve_encode_fn" in src
    assert "self._vision_encode_fn = resolve_encode_fn(self.get_logger())" in src


def test_default_tile_dims_match_projection_formula():
    """Guard: the adapter's _tile_pixel_dims default (perception off) resolves to the same
    tile size project_tiles produces from the module defaults — 480x640 at the shipped
    constants. If the tiling constants drift, this catches the CP2 gate going stale."""
    import numpy as np

    from core.perception.tiling import (
        DEFAULT_TILE_HFOV,
        DEFAULT_TILE_VFOV,
        PANO_HEIGHT,
        PANO_VFOV,
        PANO_WIDTH,
    )

    tile_w = int(round(PANO_WIDTH * DEFAULT_TILE_HFOV / (2.0 * np.pi)))
    tile_h = int(round(PANO_HEIGHT * DEFAULT_TILE_VFOV / PANO_VFOV))
    assert (tile_w, tile_h) == (480, 640)


# --------------------------------------------------------------------- real import (Ubuntu)


def test_symbols_exist_when_rclpy_present():
    pytest.importorskip("rclpy", reason="rclpy only present on Ubuntu/ROS")
    import sys

    src_root = str(_ADAPTER.parents[1])
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    import ros_adapter.adapter_node as node

    assert hasattr(node, "make_detector")
    assert hasattr(node, "_LedgerProxy")
    assert hasattr(node, "_tile_pixel_dims")
    assert node._tile_pixel_dims(None) == (480, 640)  # perception-off default tile
    assert node.DETECTOR_NONE == "none"
    # Default env -> stub detector (None).
    import os

    old = os.environ.pop(node.ENV_DETECTOR, None)
    try:
        assert node.make_detector() is None
    finally:
        if old is not None:
            os.environ[node.ENV_DETECTOR] = old
