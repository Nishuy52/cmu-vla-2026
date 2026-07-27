"""Tests for :mod:`core.calibration`.

Covers JSON round-tripping, dotted-key overrides (incl. error handling), diffing,
and — critically — that the composed defaults still match the live values in the
owning modules. The last test pins the calibration ledger to reality: if any module
constant drifts from what this ledger records, it fails loudly.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.calibration import (
    BudgetTunables,
    Calibration,
    NavTunables,
    apply_overrides,
    default_calibration,
    diff,
    from_json,
    to_json,
)


# --------------------------------------------------------------------------- round-trip


def test_default_round_trips_json_identically():
    cal = default_calibration()
    text = to_json(cal)
    restored = from_json(text)
    assert restored == cal
    # And serialisation is idempotent (deterministic key order, stable scalars).
    assert to_json(restored) == text


def test_from_json_of_overridden_round_trips():
    cal = apply_overrides(default_calibration(), {"nav.cell_m": 0.05, "budget.cap_parse": 4})
    assert from_json(to_json(cal)) == cal


def test_to_json_is_deterministic_and_parseable():
    import json

    text = to_json(default_calibration())
    parsed = json.loads(text)
    # Every declared subsystem namespace is present.
    for sub in ("geometry", "fusion", "tracker", "keyframe", "nav", "budget"):
        assert sub in parsed
    # Re-serialising the parsed structure preserves order/content.
    assert to_json(default_calibration()) == text


# --------------------------------------------------------------------------- overrides


def test_apply_overrides_changes_exactly_the_named_field():
    base = default_calibration()
    out = apply_overrides(base, {"geometry.near_floor": 1.5})
    assert out.geometry.near_floor == 1.5
    # Exactly one field changed, and it is the named one.
    assert diff(base, out) == ["geometry.near_floor"]
    # Base is untouched (frozen / copy semantics).
    assert base.geometry.near_floor == 1.2


def test_apply_overrides_multiple_dotted_paths():
    base = default_calibration()
    out = apply_overrides(
        base,
        {"geometry.near_floor": 1.5, "nav.cell_m": 0.05, "budget.watchdog_floor_s": 555.0},
    )
    assert diff(base, out) == sorted(
        ["budget.watchdog_floor_s", "geometry.near_floor", "nav.cell_m"]
    )


def test_apply_overrides_preserves_int_field_type():
    out = apply_overrides(default_calibration(), {"fusion.min_points": 9})
    assert out.fusion.min_points == 9
    assert isinstance(out.fusion.min_points, int)
    # A float literal for an int field is coerced back to int.
    out2 = apply_overrides(default_calibration(), {"nav.min_cluster_size": 7.0})
    assert out2.nav.min_cluster_size == 7
    assert isinstance(out2.nav.min_cluster_size, int)


def test_apply_overrides_unknown_field_raises():
    with pytest.raises(KeyError):
        apply_overrides(default_calibration(), {"geometry.not_a_field": 1.0})


def test_apply_overrides_unknown_subsystem_raises():
    with pytest.raises(KeyError):
        apply_overrides(default_calibration(), {"bogus.field": 1.0})


def test_apply_overrides_non_dotted_key_raises():
    with pytest.raises(KeyError):
        apply_overrides(default_calibration(), {"near_floor": 1.0})


# --------------------------------------------------------------------------- diff


def test_diff_empty_for_identical():
    assert diff(default_calibration(), default_calibration()) == []


def test_diff_detects_exactly_the_overridden_fields():
    base = default_calibration()
    changed = apply_overrides(
        base, {"tracker.gate": 1.0, "keyframe.every_k": 3, "nav.reach_m": 1.2}
    )
    assert diff(base, changed) == sorted(["keyframe.every_k", "nav.reach_m", "tracker.gate"])


# --------------------------------------------------------------------------- pinning to live modules


def test_composed_defaults_match_live_geometry_perception_configs():
    """The composed config dataclasses must BE the live defaults, not copies."""
    from core.geometry.toolbox import DEFAULT_THRESHOLDS
    from core.perception.fusion import DEFAULT_FUSION_CONFIG
    from core.perception.tracker import DEFAULT_KEYFRAME_CONFIG, DEFAULT_TRACKER_CONFIG

    cal = default_calibration()
    assert cal.geometry == DEFAULT_THRESHOLDS
    assert cal.fusion == DEFAULT_FUSION_CONFIG
    assert cal.tracker == DEFAULT_TRACKER_CONFIG
    assert cal.keyframe == DEFAULT_KEYFRAME_CONFIG


def test_nav_tunables_match_live_module_constants():
    """NavTunables defaults must equal the loose nav module-level constants."""
    from core.nav import breadcrumbs, costmap, exploration, frontiers, occupancy, planner

    nav = NavTunables()
    # occupancy.py
    assert nav.cell_m == occupancy.CELL_M
    assert nav.free_max == occupancy.FREE_MAX
    assert nav.observe_radius_m == occupancy.OBSERVE_RADIUS_M
    assert nav.grow_pad_cells == occupancy.GROW_PAD_CELLS
    # costmap.py
    assert nav.vehicle_radius_m == costmap.VEHICLE_RADIUS_M
    assert nav.overhead_soft_cost_mult == costmap.OVERHEAD_SOFT_COST_MULT
    # frontiers.py
    assert nav.min_cluster_size == frontiers.MIN_CLUSTER_SIZE
    assert nav.w_size == frontiers.W_SIZE
    assert nav.w_dist == frontiers.W_DIST
    assert nav.w_affinity == frontiers.W_AFFINITY
    # planner.py
    assert nav.unknown_cost_mult == planner.UNKNOWN_COST_MULT
    assert nav.pinch_disc_m == planner.PINCH_DISC_M
    assert nav.pinch_corridor_half_w_m == planner.PINCH_CORRIDOR_HALF_W_M
    # exploration.py
    assert nav.sweep_s == exploration.SWEEP_S
    assert nav.sweep_side_m == exploration.SWEEP_SIDE_M
    assert nav.min_frontier_score == exploration.MIN_FRONTIER_SCORE
    assert nav.coverage_saturated_free_frac == exploration.COVERAGE_SATURATED_FREE_FRAC
    # breadcrumbs.py
    assert nav.lookahead_m == breadcrumbs.LOOKAHEAD_M
    assert nav.reach_m == breadcrumbs.REACH_M
    assert nav.stall_move_m == breadcrumbs.STALL_MOVE_M
    assert nav.stall_window_s == breadcrumbs.STALL_WINDOW_S


def test_budget_tunables_match_live_module_constants():
    """BudgetTunables defaults must equal the fsm/budget + interfaces constants."""
    from core.fsm import budget as budget_mod
    from core import interfaces
    from core.interfaces import (
        EXPLORE_BUDGET_S,
        FORCED_ASSEMBLY_S,
        QUESTION_BUDGET_S,
        QType,
        TerrainPatch,
        WATCHDOG_FLOOR_S,
    )

    bt = BudgetTunables()
    assert bt.question_budget_s == QUESTION_BUDGET_S
    assert bt.forced_assembly_s == FORCED_ASSEMBLY_S
    assert bt.watchdog_floor_s == WATCHDOG_FLOOR_S
    assert bt.terrain_free_max == TerrainPatch.FREE_MAX
    assert bt.orientation_s == budget_mod.ORIENTATION_S
    assert bt.ledger_reserve_s == budget_mod.LEDGER_RESERVE_S
    # per-QType exploration budgets reconstruct the live mapping exactly
    assert bt.explore_budget_map() == EXPLORE_BUDGET_S
    # per-checkpoint caps reconstruct the live mapping exactly
    assert bt.checkpoint_max_map() == budget_mod.CHECKPOINT_MAX


def test_budget_reconstructed_maps_have_no_missing_or_extra_keys():
    """Guard against a QType/checkpoint being added to the module but not the ledger."""
    from core.fsm.budget import CHECKPOINT_MAX
    from core.interfaces import EXPLORE_BUDGET_S

    bt = BudgetTunables()
    assert set(bt.explore_budget_map()) == set(EXPLORE_BUDGET_S)
    assert set(bt.checkpoint_max_map()) == set(CHECKPOINT_MAX)


def test_field_counts_per_subsystem():
    """Sanity pin on the ledger size so a dropped field is noticed."""
    from dataclasses import fields

    cal = default_calibration()
    counts = {
        "geometry": len(fields(cal.geometry)),
        "fusion": len(fields(cal.fusion)),
        "tracker": len(fields(cal.tracker)),
        "keyframe": len(fields(cal.keyframe)),
        "nav": len(fields(cal.nav)),
        "budget": len(fields(cal.budget)),
    }
    assert counts == {
        "geometry": 18,  # +3 colour-salience knobs (issues #11/#12)
        "fusion": 5,
        "tracker": 5,  # +3 size-aware association knobs (issues #94/#89)
        "keyframe": 3,
        "nav": 21,
        "budget": 15,
    }


def test_numpy_scalar_fields_serialise_as_plain_floats():
    """np.deg2rad-derived fields must survive JSON as plain floats (round-trip safe)."""
    import json

    cal = default_calibration()
    parsed = json.loads(to_json(cal))
    # fusion.angular_pad and keyframe.min_rotation come from np.deg2rad.
    assert isinstance(parsed["fusion"]["angular_pad"], float)
    assert isinstance(parsed["keyframe"]["min_rotation"], float)
    assert parsed["fusion"]["angular_pad"] == pytest.approx(float(np.deg2rad(3.0)))
