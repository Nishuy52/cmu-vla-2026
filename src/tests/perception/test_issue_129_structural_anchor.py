"""Issue #129 acceptance gate: structural classes leave the census but keep working
as relation anchors.

`window` is structural (core.perception.vocab.is_structural_class) but is the 4th
most frequent object head across the training question set and is used almost
entirely as a spatial ANCHOR ("how many sofas are below a window?",
docs/question_analysis.md). This module proves end-to-end, through the SAME
production entry points the real answer heads use
(core.geometry.toolbox.counting/resolve), that a window anchor still resolves a
relation clause after structural classes were excluded from the index's countable
census -- the gate the #129 fix must not break.
"""
from __future__ import annotations

import numpy as np

from core.geometry.toolbox import counting, resolve
from core.interfaces import InstanceRecord
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, Clause, Pred, TargetSpec


def _rec(instance_id, label, cmin, cmax, n_obs=3, score=0.9) -> InstanceRecord:
    amin = np.asarray(cmin, dtype=float)
    amax = np.asarray(cmax, dtype=float)
    return InstanceRecord(
        instance_id=instance_id,
        label=label,
        score=score,
        n_obs=n_obs,
        centroid=(amin + amax) / 2.0,
        aabb_min=amin,
        aabb_max=amax,
    )


def _scene() -> BasicSceneIndex:
    return BasicSceneIndex(
        [
            # a window on the wall (structural)
            _rec(1, "window", [0.0, 0.0, 1.0], [1.0, 0.1, 2.0]),
            # a sofa near the window -- the true target of "the sofa near a window"
            _rec(2, "sofa", [0.2, 0.5, 0.0], [1.2, 1.3, 0.8]),
            # a sofa far from the window -- must NOT count/resolve as near it
            _rec(3, "sofa", [10.0, 10.0, 0.0], [11.0, 10.8, 0.8]),
            # floor fragments -- structural, over-tracked (#129's own defect); must
            # not leak into the sofa/window resolution above
            _rec(4, "floor", [0, 0, 0], [1, 1, 0.05]),
            _rec(5, "floor", [5, 5, 0], [6, 6, 0.05]),
        ]
    )


def _near_window_target() -> TargetSpec:
    return TargetSpec(
        noun="sofa",
        clauses=[Clause(pred=Pred.NEAR, anchors=[Anchor(noun="window")])],
    )


def test_window_anchor_still_resolves_a_relation_clause():
    index = _scene()
    result = resolve(_near_window_target(), index)
    assert result.candidates_ranked, "window anchor failed to resolve any candidate"
    assert result.candidates_ranked[0].instance_id == 2  # the sofa actually near it
    # the relaxation ladder must not have needed to fall back to category-only --
    # a genuinely resolved anchor, not a guess.
    assert not any(r.kind == "category_only" for r in result.audit)


def test_how_many_sofas_near_a_window_counts_via_the_anchor():
    index = _scene()
    result = counting(_near_window_target(), index)
    assert result.count == 1
    assert result.ids == {2}


def test_window_anchor_unaffected_by_floor_over_tracking():
    # The defect (#129): floor is proposed/tracked as many instances (here 2, live
    # scenes show up to 20). Neither the window anchor's own resolution nor the
    # sofa/window relation it gates may be perturbed by that -- floor never
    # participates in this clause at all.
    index = _scene()
    baseline = counting(_near_window_target(), index)
    # add a THIRD floor fragment (further inflating the defect) and confirm the
    # sofa/window answer is completely unchanged.
    index.add(_rec(6, "floor", [8, 8, 0], [9, 9, 0.05]))
    after = counting(_near_window_target(), index)
    assert after.count == baseline.count == 1
    assert after.ids == baseline.ids == {2}


def test_structural_anchor_excluded_from_its_own_class_census_but_still_an_anchor():
    # The two-sided acceptance gate in one test: `window` leaves the countable
    # census (countable_instances) while still being a fully working anchor
    # (by_label/resolve/counting, exercised above and here again via by_label).
    index = _scene()
    assert "window" not in {r.label for r in index.countable_instances()}
    assert len(index.by_label("window")) == 1
    assert len(index.by_label("floor")) == 2
    assert "floor" not in {r.label for r in index.countable_instances()}
