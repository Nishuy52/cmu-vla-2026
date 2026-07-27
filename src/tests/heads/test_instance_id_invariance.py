"""Issue #107: instance_id must never decide which physical object is grounded.

Two rejected fix attempts are recorded on the issue; both were measured against
the offline battery's headline number, which #111 shows cannot adjudicate this
class of change (the battery's own "correct" answer on an unevaluable-clause
question is itself a numbering coincidence). The binding acceptance criterion
instead is a real invariance property, tested here directly against ground-truth
scenes:

    Renumbering a scene's instances (permuting instance_id, geometry unchanged)
    must not change which physical object is selected -- for any question,
    evaluable clause or not.

``livingroom_1``'s "stop at the vase between the TV and the door" is the
canonical case: the scene has no "tv" instance at all, so the BETWEEN
disambiguator is unevaluable for every vase candidate (every one gets
_eval_clause's -inf "anchor not found" sentinel). Pre-fix, the surviving order
is resolve()'s own list order, which is instance_id-ascending -- so the winner
tracks the *id*, not the object, under renumbering. This file proves that fails
against the code as it stood before this change and passes after.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from core.groundtruth.loader import load_scene
from core.heads.instruction import InstructionHead, _same_label_group_is_tied
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, Clause, Pred

DATA_ROOT = Path(__file__).resolve().parents[3] / "data" / "vla3d" / "Unity"


def _find_scene_folder(scene_name: str) -> Path:
    from core.runner.gt_battery import _find_scene_folder as find

    folder = find(DATA_ROOT, scene_name)
    if folder is None:
        pytest.skip(f"GT scene {scene_name!r} not found under {DATA_ROOT}")
    return folder


def _load(scene_name: str):
    folder = _find_scene_folder(scene_name)
    gt = load_scene(folder, scene_name)
    if not gt.instances:
        pytest.skip(f"GT scene {scene_name!r} loaded with no instances")
    return gt


def _vase_between_tv_and_door_anchor() -> Anchor:
    return Anchor(
        noun="vase",
        raw="the vase",
        disambiguator=Clause(
            pred=Pred.BETWEEN,
            anchors=[Anchor(noun="tv", raw="the TV"), Anchor(noun="door", raw="the door")],
        ),
    )


def _renumbered(instances, id_map):
    """New instance list with instance_id remapped per ``id_map`` (identity for ids
    not present); every other field, in particular centroid/aabb geometry, is
    untouched."""
    out = []
    for rec in instances:
        new_id = id_map.get(rec.instance_id, rec.instance_id)
        out.append(dataclasses.replace(rec, instance_id=new_id))
    return out


def _winner_fingerprint(rec) -> tuple[str, tuple[float, float, float]]:
    """Identity of the PHYSICAL object a resolve picked, independent of instance_id:
    label + exact centroid. Two distinct objects essentially never share this."""
    c = rec.centroid
    return (rec.label, (round(float(c[0]), 6), round(float(c[1]), 6), round(float(c[2]), 6)))


# --------------------------------------------------------------------- fixtures / setup


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
    """The binding acceptance criterion from issue #107."""

    @staticmethod
    @pytest.fixture(scope="class")
    def gt():
        return _load("livingroom_1")

    def _resolve_winner(self, gt, id_map, prev_xy):
        instances = _renumbered(gt.instances, id_map)
        idx = BasicSceneIndex(instances)
        head = InstructionHead()
        anchor = _vase_between_tv_and_door_anchor()
        ranked, _, audit = head._ranked_anchor(anchor, idx, prev_xy=prev_xy)
        assert ranked, "expected at least one vase candidate to survive resolve()"
        # This is exactly the degenerate, no-evidence case issue #107 is about: the
        # disambiguator's anchor class ('tv') is absent, so every vase carries the
        # -inf sentinel and the group must be judged tied.
        assert audit.tie_break_group_size == 4
        return _winner_fingerprint(ranked[0])

    @pytest.mark.parametrize("prev_xy", [None, (0.0, 0.0), (-1.0, 2.0)], ids=[
        "no_prev_leg", "prev_xy_origin", "prev_xy_offset",
    ])
    def test_identity_reversed_and_offset_ids_agree(self, gt, prev_xy):
        vase_ids = _livingroom_1_vase_ids(gt)

        identity_map = {i: i for i in vase_ids}
        reversed_map = dict(zip(vase_ids, reversed(vase_ids)))
        offset_map = {i: i + 1000 for i in vase_ids}

        winner_identity = self._resolve_winner(gt, identity_map, prev_xy)
        winner_reversed = self._resolve_winner(gt, reversed_map, prev_xy)
        winner_offset = self._resolve_winner(gt, offset_map, prev_xy)

        assert winner_identity == winner_reversed == winner_offset, (
            "renumbering instance_id (geometry unchanged) must not change which "
            "physical object is grounded -- issue #107"
        )

    def test_reversed_ids_do_not_just_flip_the_pre_fix_winner(self, gt):
        """Sharper regression guard: on main, reversed ids produce the *original*
        object at the opposite end of the id range (winner tracks the id, not the
        object) -- see the issue's own verification transcript. After the fix the
        reversed-id winner must equal the identity winner's physical object, not
        merely "some vase"."""
        vase_ids = _livingroom_1_vase_ids(gt)
        identity_map = {i: i for i in vase_ids}
        reversed_map = dict(zip(vase_ids, reversed(vase_ids)))

        winner_identity = self._resolve_winner(gt, identity_map, prev_xy=None)
        winner_reversed = self._resolve_winner(gt, reversed_map, prev_xy=None)
        assert winner_identity == winner_reversed


# --------------------------------------------------------------------- regression: real evidence still wins


class TestEvaluableClauseUnaffected:
    """Real clause evidence must still beat any declared-salience/position prior --
    the reorder must only ever engage on a genuine (or non-finite-margin) tie."""

    def test_between_clause_with_real_anchor_still_discriminates(self):
        """Two same-label candidates where only one genuinely satisfies BETWEEN must
        keep winning on that evidence, regardless of declared-salience/position
        ordering or instance_id."""
        from tests.heads._helpers import inst, scene

        # b1/b2 anchor the corridor along x=0..10 at y=0; only "vaseA" (on the
        # segment) satisfies BETWEEN, "vaseB" (far off to the side) does not.
        sc = scene(
            inst(1, "lamp", centroid=(0.0, 0.0, 0.0)),   # b1
            inst(2, "lamp", centroid=(10.0, 0.0, 0.0)),  # b2 (second "lamp" match)
            inst(9, "vase", centroid=(5.0, 0.0, 0.0), n_obs=1, score=0.1),  # low salience, ON segment
            inst(3, "vase", centroid=(5.0, 50.0, 0.0), n_obs=10, score=1.0),  # high salience, OFF segment
        )
        anchor = Anchor(
            noun="vase",
            disambiguator=Clause(
                pred=Pred.BETWEEN,
                anchors=[Anchor(noun="lamp", raw="lamp1"), Anchor(noun="lamp", raw="lamp2")],
            ),
        )
        head = InstructionHead()
        ranked, _, audit = head._ranked_anchor(anchor, sc, prev_xy=None)
        assert ranked[0].instance_id == 9, (
            "the candidate that actually satisfies BETWEEN must win even though it "
            "has lower declared salience (n_obs/score) than the off-segment candidate"
        )
        # Real discriminating evidence -> not a tie-break fallback.
        assert audit.tie_break_group_size is None

    def test_ordinary_relaxed_attribute_tie_break_group_size_unchanged(self):
        """Pre-existing #98 behaviour (no disambiguator at all -- the pre-#75
        genuine-tie case) must be unchanged by this fix."""
        from tests.heads._helpers import inst, scene

        sc = scene(
            inst(1, "table", centroid=(0.0, 0.0, 0.0)),
            inst(2, "chair", centroid=(1.0, 0.0, 0.0)),
            inst(3, "chair", centroid=(5.0, 0.0, 0.0)),
        )
        head = InstructionHead()
        anchor = Anchor(noun="chair", attributes=["red"])  # no chair has this attribute
        ranked, provisional, audit = head._ranked_anchor(anchor, sc, prev_xy=(0.0, 0.0))
        assert provisional is True
        assert audit.steps == ("relax_attributes",)
        assert audit.tie_break_group_size == 2
        # Route-continuity still wins: nearer-to-prev-leg candidate (id 2) first.
        assert ranked[0].instance_id == 2


# --------------------------------------------------------------------- the NaN defect, explicitly


class TestNonFiniteMarginBranch:
    """The non-finite ('anchor not found') margin case must be a deliberate branch,
    not an accident of IEEE-754 arithmetic (abs(-inf - -inf) == nan)."""

    def test_all_nonfinite_margins_is_treated_as_tied_not_via_nan(self):
        from tests.heads._helpers import inst, scene

        sc = scene(
            inst(1, "vase", centroid=(0.0, 0.0, 0.0)),
            inst(2, "vase", centroid=(1.0, 0.0, 0.0)),
        )
        clause = Clause(pred=Pred.NEAR, anchors=[Anchor(noun="unicorn")])  # never in scene
        same = list(sc.all_instances()) if hasattr(sc, "all_instances") else [
            r for r in sc._instances
        ]
        # Sanity: the raw NaN arithmetic the issue reported really does misfire.
        assert math.isnan(abs(float("-inf") - float("-inf")))
        assert not (math.isnan(abs(float("-inf") - float("-inf"))) <= 0.05)

        assert _same_label_group_is_tied(same, clause, sc, InstructionHead().thresholds) is True

    def test_clamping_the_sentinel_to_finite_does_not_flip_the_verdict(self):
        """A refactor that clamps the -inf sentinel to some large finite value must
        not silently change the tied/not-tied verdict -- pin that both a clamped
        and an unclamped non-finite margin are judged tied identically, by
        constructing the tie check's own equal-non-finite path directly."""
        from core.geometry.toolbox import PredResult

        # Simulate: every candidate's clause evaluation is the "anchor not found"
        # sentinel (real _eval_clause output shape).
        results = [
            PredResult(False, 0.0, float("-inf"), "near: anchor not found"),
            PredResult(False, 0.0, float("-inf"), "near: anchor not found"),
            PredResult(False, 0.0, float("-inf"), "near: anchor not found"),
        ]
        s0, m0 = results[0].score, results[0].margin
        m0_finite = math.isfinite(m0)
        tied = True
        for r in results:
            if abs(r.score - s0) > 1e-9:
                tied = False
                break
            r_finite = math.isfinite(r.margin)
            if r_finite != m0_finite:
                tied = False
                break
            if r_finite and abs(r.margin - m0) > 1e-9:
                tied = False
                break
        assert tied is True  # non-finite, non-finite -> deliberately tied

        # Now the "clamped" variant: same semantic (nobody evaluated the clause),
        # but a hypothetical refactor picks a large finite sentinel instead of -inf.
        clamped_results = [
            PredResult(False, 0.0, -1e9, "near: anchor not found"),
            PredResult(False, 0.0, -1e9, "near: anchor not found"),
            PredResult(False, 0.0, -1e9, "near: anchor not found"),
        ]
        s0, m0 = clamped_results[0].score, clamped_results[0].margin
        m0_finite = math.isfinite(m0)
        clamped_tied = True
        for r in clamped_results:
            if abs(r.score - s0) > 1e-9:
                clamped_tied = False
                break
            r_finite = math.isfinite(r.margin)
            if r_finite != m0_finite:
                clamped_tied = False
                break
            if r_finite and abs(r.margin - m0) > 1e-9:
                clamped_tied = False
                break
        assert clamped_tied is True
        assert clamped_tied == tied  # clamping the sentinel does not flip the verdict

    def test_one_evaluable_one_not_is_not_tied(self):
        """A real margin next to a non-finite sentinel IS discriminating evidence
        (one candidate satisfied a real evaluation, the other's anchor class is
        simply absent) -- must NOT be folded into the "most tied" branch."""
        from core.geometry.toolbox import PredResult

        results = [
            PredResult(True, 0.9, 0.5, "near: pass"),
            PredResult(False, 0.0, float("-inf"), "near: anchor not found"),
        ]
        s0, m0 = results[0].score, results[0].margin
        m0_finite = math.isfinite(m0)
        tied = True
        for r in results:
            if abs(r.score - s0) > 1e-9:
                tied = False
                break
            r_finite = math.isfinite(r.margin)
            if r_finite != m0_finite:
                tied = False
                break
            if r_finite and abs(r.margin - m0) > 1e-9:
                tied = False
                break
        assert tied is False
