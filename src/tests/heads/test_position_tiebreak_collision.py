"""Issue #116: the quantised-position final tie-break (#107/#113) must be
observably guarded against the degenerate case where two DISTINCT same-label
candidates collide at millimetre precision on both axes.

``_quantized_position`` rounds each candidate's (x, y) centroid to 3 decimal
places. If two same-label candidates' centroids agree to within 1 mm on both
axes, the full sort key still ties after that quantisation, and Python's
stable sort silently preserves the candidates' relative pre-sort order --
which traces back to resolve()'s own candidate order, exactly the
instance_id-correlated artifact #107/#111/#113 were meant to eliminate.

This is a detection-only guard: it must never change which candidate wins.
"""
from __future__ import annotations

import logging

import pytest

from core.heads.instruction import InstructionHead
from core.plan_schema import Anchor
from tests.heads._helpers import inst, scene


def _colliding_vase_scene():
    """Two same-label "vase" candidates whose centroids differ by < 1 mm on
    both x and y, but round to the SAME (x, y) at 3 dp -- and whose n_obs /
    score / extent are identical, so _declared_salience_key also ties. Only
    _quantized_position is left to discriminate them, and it can't."""
    return scene(
        inst(1, "vase", centroid=(1.0001, 2.0001, 0.0)),
        inst(2, "vase", centroid=(1.0004, 2.0004, 0.0)),
    )


class TestPositionTiebreakCollisionGuard:
    def test_collision_sets_audit_flag_and_logs_warning_no_prev_leg(self, caplog):
        sc = _colliding_vase_scene()
        anchor = Anchor(noun="vase")
        head = InstructionHead()
        with caplog.at_level(logging.WARNING, logger="core.heads.instruction"):
            ranked, _, audit = head._ranked_anchor(anchor, sc, prev_xy=None)
        assert len(ranked) == 2
        assert audit.tie_break_group_size == 2
        assert audit.position_collision is True, (
            "millimetre centroid collision must set the #116 guard flag"
        )
        assert any("quantised-position collision" in r.message for r in caplog.records), (
            "the collision must also be observable as a logged warning"
        )

    def test_collision_sets_audit_flag_with_prev_leg(self, caplog):
        # With a previous leg, the sort key's FIRST component is raw squared
        # distance to prev_xy -- which only ties, in general, when the raw
        # (unquantized) centroids are identical, not merely within rounding
        # distance of each other. Two distinct instances sitting at the exact
        # same point (0 mm apart, well within the 1 mm bound) is itself a
        # legitimate instance of the collision (cf. #112's over-segmentation:
        # duplicate instances of one physical object land at the same centroid).
        sc = scene(
            inst(1, "vase", centroid=(3.0, 4.0, 0.0)),
            inst(2, "vase", centroid=(3.0, 4.0, 0.0)),
        )
        anchor = Anchor(noun="vase")
        head = InstructionHead()
        with caplog.at_level(logging.WARNING, logger="core.heads.instruction"):
            ranked, _, audit = head._ranked_anchor(anchor, sc, prev_xy=(0.0, 0.0))
        assert len(ranked) == 2
        assert audit.position_collision is True
        assert any("quantised-position collision" in r.message for r in caplog.records)

    def test_non_colliding_group_never_sets_the_flag(self):
        """Sanity/negative control: candidates that are actually distinguishable by
        quantised position must NOT trip the guard (issue #98's ordinary case)."""
        sc = scene(
            inst(1, "table", centroid=(0.0, 0.0, 0.0)),
            inst(2, "chair", centroid=(1.0, 0.0, 0.0)),
            inst(3, "chair", centroid=(5.0, 0.0, 0.0)),
        )
        head = InstructionHead()
        anchor = Anchor(noun="chair", attributes=["red"])  # no chair has this attribute
        _, _, audit = head._ranked_anchor(anchor, sc, prev_xy=(0.0, 0.0))
        assert audit.tie_break_group_size == 2
        assert audit.position_collision is False

    def test_winner_is_unaffected_by_the_guard(self):
        """The guard is detection-only: the winning candidate must be exactly the
        one Python's stable sort over the (unchanged) key would already pick --
        i.e. adding the guard must not perturb ``same``'s order."""
        sc = _colliding_vase_scene()
        anchor = Anchor(noun="vase")
        head = InstructionHead()
        ranked, _, audit = head._ranked_anchor(anchor, sc, prev_xy=None)
        assert audit.position_collision is True
        # Both candidates tie on every key component, so stable sort preserves
        # resolve()'s own pre-sort order -- id 1 was constructed first and
        # resolve() ranks it first among equals.
        assert ranked[0].instance_id == 1
