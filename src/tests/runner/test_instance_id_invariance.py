"""Issue #113: gt_battery's own copy of the same-label salience reorder must
never let ``instance_id`` decide which physical object is grounded.

Mirrors issue #107's head-side fix (see
``tests/heads/test_instance_id_invariance.py``) but exercises
``core.runner.gt_battery`` directly -- the battery carried its own independent
copy of the reorder (``_if_rubric_geometry``'s nested ``_resolve_anchor_rec``)
with the exact same two defects: the sort key's final tie-break was
``instance_id``, and ``_same_label_group_is_tied`` compared ``nan`` when every
candidate's margin was the ``-inf`` "anchor not found" sentinel.

``livingroom_1``'s "stop at the vase between the TV and the door" (from the
official training questions, ``upstream/CMU-VLN-Challenge-2026/questions/
questions.json``) is the canonical case: the scene has no "tv" instance at
all, so the BETWEEN disambiguator is unevaluable for every vase candidate.
Pre-fix, the surviving order is ``resolve()``'s own list order, which is
instance_id-ascending -- so the winner tracks the *id*, not the object, under
renumbering. This file proves that fails against the code as it stood before
this change and passes after.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from core.groundtruth.loader import load_scene
from core.perception.scene_index import BasicSceneIndex
from core.runner.gt_battery import (
    _find_scene_folder,
    _if_rubric_geometry,
    _same_label_group_is_tied,
)

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "vla3d" / "Unity"

#: The exact training instruction (upstream questions.json, livingroom_1,
#: instruction_following[0]) whose second leg's disambiguator anchor ("tv") is
#: absent from the scene's GT (only "tv cabinet"/"tv remote" are annotated --
#: see issue #110), making BETWEEN(tv, door) unevaluable for every vase.
LIVINGROOM_1_TEXT = (
    "Go to the potted plant closest to the pyramid candle holder and stop "
    "at the vase between the TV and the door."
)


def _load(scene_name: str):
    folder = _find_scene_folder(DATA_ROOT, scene_name)
    if folder is None:
        pytest.skip(f"GT scene {scene_name!r} not found under {DATA_ROOT}")
    gt = load_scene(folder, scene_name)
    if not gt.instances:
        pytest.skip(f"GT scene {scene_name!r} loaded with no instances")
    return gt


def _renumbered(instances, id_map):
    """New instance list with instance_id remapped per ``id_map`` (identity for
    ids not present); every other field, in particular centroid/aabb geometry,
    is untouched."""
    out = []
    for rec in instances:
        new_id = id_map.get(rec.instance_id, rec.instance_id)
        out.append(dataclasses.replace(rec, instance_id=new_id))
    return out


def _fingerprint(rec) -> tuple[str, tuple[float, float, float]]:
    """Identity of the PHYSICAL object a resolve picked, independent of
    instance_id: label + exact centroid. Two distinct objects essentially
    never share this."""
    c = rec.centroid
    return (rec.label, (round(float(c[0]), 6), round(float(c[1]), 6), round(float(c[2]), 6)))


def _livingroom_1_vase_ids(gt) -> list[int]:
    ids = sorted(r.instance_id for r in gt.instances if r.label == "vase")
    assert len(ids) == 4, f"expected 4 vase instances in livingroom_1, found {ids}"
    assert "tv" not in {r.label for r in gt.instances}, (
        "test assumes livingroom_1 has no 'tv' instance -- this is what makes the "
        "BETWEEN(tv, door) disambiguator unevaluable and every vase carry the -inf "
        "sentinel margin; if the scene now has a 'tv', pick a different scene/case"
    )
    return ids


# --------------------------------------------------------------------- invariance tests


class TestRenumberingInvariance:
    """The binding acceptance criterion from issue #113 (mirrors #107)."""

    @staticmethod
    @pytest.fixture(scope="class")
    def gt():
        return _load("livingroom_1")

    def _resolve_vase_winner(self, gt, id_map):
        instances = _renumbered(gt.instances, id_map)
        idx = BasicSceneIndex(instances)
        instances_by_id = {r.instance_id: r for r in instances}
        leg_goals, _, _, leg_instance_ids, _ = _if_rubric_geometry(
            LIVINGROOM_1_TEXT, gt, idx, start_xy=None
        )
        assert len(leg_goals) == 2, (
            "expected both legs (potted plant, vase) to resolve; got "
            f"{leg_goals!r} -- scene contents may have changed"
        )
        vase_ids = leg_instance_ids[-1]
        assert vase_ids is not None and len(vase_ids) == 1
        return _fingerprint(instances_by_id[vase_ids[0]])

    def test_identity_reversed_and_offset_ids_agree(self, gt):
        vase_ids = _livingroom_1_vase_ids(gt)

        identity_map = {i: i for i in vase_ids}
        reversed_map = dict(zip(vase_ids, reversed(vase_ids)))
        offset_map = {i: i + 1000 for i in vase_ids}

        winner_identity = self._resolve_vase_winner(gt, identity_map)
        winner_reversed = self._resolve_vase_winner(gt, reversed_map)
        winner_offset = self._resolve_vase_winner(gt, offset_map)

        assert winner_identity == winner_reversed == winner_offset, (
            "renumbering instance_id (geometry unchanged) must not change which "
            "physical object the battery resolves -- issue #113"
        )

    def test_reversed_ids_do_not_just_flip_the_pre_fix_winner(self, gt):
        """Sharper regression guard: on main, reversed ids produce the *original*
        object at the opposite end of the id range (winner tracks the id, not
        the object) -- see the issue's own verification transcript. After the
        fix the reversed-id winner must equal the identity winner's physical
        object, not merely "some vase"."""
        vase_ids = _livingroom_1_vase_ids(gt)
        identity_map = {i: i for i in vase_ids}
        reversed_map = dict(zip(vase_ids, reversed(vase_ids)))

        winner_identity = self._resolve_vase_winner(gt, identity_map)
        winner_reversed = self._resolve_vase_winner(gt, reversed_map)
        assert winner_identity == winner_reversed


# --------------------------------------------------------------------- regression: real evidence still wins


class TestEvaluableClauseUnaffected:
    """Real clause evidence must still beat any declared-salience/position prior
    -- the reorder must only ever engage on a genuine (or non-finite-margin)
    tie."""

    def test_between_clause_with_real_anchor_still_discriminates(self):
        """Two same-label candidates where only one genuinely satisfies BETWEEN
        must be judged NOT tied on that evidence -- the group's ``margin``s
        differ, so the salience/position reorder must not engage and
        ``resolve()``'s own evidence-based order must survive."""
        from core.geometry.toolbox import DEFAULT_THRESHOLDS, _eval_clause
        from core.plan_schema import Anchor, Clause, Pred
        from tests.heads._helpers import inst

        # b1/b2 anchor the corridor along x=0..10 at y=0; only "vaseA" (on the
        # segment) satisfies BETWEEN, "vaseB" (far off to the side) does not.
        lamp1 = inst(1, "lamp", centroid=(0.0, 0.0, 0.0))
        lamp2 = inst(2, "lamp", centroid=(10.0, 0.0, 0.0))
        vase_on = inst(9, "vase", centroid=(5.0, 0.0, 0.0), n_obs=1, score=0.1)
        vase_off = inst(3, "vase", centroid=(5.0, 50.0, 0.0), n_obs=10, score=1.0)
        from core.perception.scene_index import BasicSceneIndex

        sc = BasicSceneIndex([lamp1, lamp2, vase_on, vase_off])
        disamb = Clause(
            pred=Pred.BETWEEN,
            anchors=[Anchor(noun="lamp", raw="lamp1"), Anchor(noun="lamp", raw="lamp2")],
        )
        same = [vase_on, vase_off]
        tied = _same_label_group_is_tied(same, disamb, sc, DEFAULT_THRESHOLDS, _eval_clause)
        assert tied is False, (
            "the two vase candidates carry different BETWEEN margins (one on-"
            "segment, one off) -- this is real discriminating evidence, so the "
            "group must NOT be judged tied and the reorder must not engage"
        )

    def test_ordinary_relaxed_attribute_tie_uses_route_continuity(self):
        """Pre-existing behaviour (no disambiguator at all -- the pre-#75
        genuine-tie case) still route-continuity-orders on ``approach_xy`` when
        one is available."""
        from core.plan_schema import Anchor, TargetSpec
        from core.geometry.toolbox import resolve
        from tests.heads._helpers import inst, scene

        sc = scene(
            inst(1, "table", centroid=(0.0, 0.0, 0.0)),
            inst(2, "chair", centroid=(1.0, 0.0, 0.0)),
            inst(3, "chair", centroid=(5.0, 0.0, 0.0)),
        )
        spec = TargetSpec(noun="chair", raw="the red chair", attributes=["red"], clauses=[])
        res = resolve(spec, sc)
        # Sanity: both chairs survive to a tie (no disambiguator clause here).
        same = [c for c in res.candidates_ranked if c.label == "chair"]
        assert len(same) == 2
        assert _same_label_group_is_tied(same, None, sc, None, None) is True


# --------------------------------------------------------------------- the NaN defect, explicitly


class TestNonFiniteMarginBranch:
    """The non-finite ('anchor not found') margin case must be a deliberate
    branch, not an accident of IEEE-754 arithmetic (``abs(-inf - -inf) ==
    nan``)."""

    def test_all_nonfinite_margins_is_treated_as_tied_not_via_nan(self):
        from core.geometry.toolbox import PredResult

        # Sanity: the raw NaN arithmetic the issue reported really does misfire.
        assert math.isnan(abs(float("-inf") - float("-inf")))
        assert not (math.isnan(abs(float("-inf") - float("-inf"))) <= 0.05)

        same = ["a", "b", "c"]  # opaque candidates -- eval_clause below ignores them

        def eval_clause(_c, _clause, _idx, _th):
            return PredResult(False, 0.0, float("-inf"), "between: anchor not found")

        assert _same_label_group_is_tied(same, object(), None, None, eval_clause) is True

    def test_clamping_the_sentinel_to_finite_does_not_flip_the_verdict(self):
        """A refactor that clamps the -inf sentinel to some large finite value
        must not silently change the tied/not-tied verdict -- both an
        unclamped and a clamped non-finite margin must be judged tied
        identically through the real ``_same_label_group_is_tied`` entry
        point."""
        from core.geometry.toolbox import PredResult

        same = ["a", "b", "c"]

        def eval_clause_inf(_c, _clause, _idx, _th):
            return PredResult(False, 0.0, float("-inf"), "between: anchor not found")

        def eval_clause_clamped(_c, _clause, _idx, _th):
            return PredResult(False, 0.0, -1e9, "between: anchor not found")

        tied_inf = _same_label_group_is_tied(same, object(), None, None, eval_clause_inf)
        tied_clamped = _same_label_group_is_tied(same, object(), None, None, eval_clause_clamped)
        assert tied_inf is True
        assert tied_clamped is True
        assert tied_inf == tied_clamped

    def test_one_evaluable_one_not_is_not_tied(self):
        """A real margin next to a non-finite sentinel IS discriminating
        evidence (one candidate satisfied a real evaluation, the other's
        anchor class is simply absent) -- must NOT be folded into the "most
        tied" branch."""
        from core.geometry.toolbox import PredResult

        same = ["a", "b"]
        results = iter(
            [
                PredResult(True, 0.9, 0.5, "between: pass"),
                PredResult(False, 0.0, float("-inf"), "between: anchor not found"),
            ]
        )

        def eval_clause(_c, _clause, _idx, _th):
            return next(results)

        assert _same_label_group_is_tied(same, object(), None, None, eval_clause) is False
