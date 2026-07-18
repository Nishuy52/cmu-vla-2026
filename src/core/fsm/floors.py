"""Always-ready degraded answers — the watchdog's guaranteed-legal fallback per type.

"Silence is the only unforgivable failure" (architecture §1 row 8): a floor answer for
every qtype is recomputed continuously from whatever the map currently holds, so a
publishable answer is available at any tick regardless of pipeline state.

get(qtype) returns:
  NUMERICAL             -> IntAnswer
  OBJECT_REFERENCE      -> MarkerBox
  INSTRUCTION_FOLLOWING -> WaypointCmd

update() and get() are total functions: they never raise, even with an empty scene,
a None plan, or malformed partial results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.interfaces import (
    InstanceRecord,
    IntAnswer,
    MarkerBox,
    QType,
    SceneIndex,
    WaypointCmd,
)
from core.perception.detector import is_answer_eligible
from core.perception.dimension_priors import clamp_record_marker

MODAL_COUNT: int = 2  # most-common integer answer in the training distribution (architecture §4)
_UNIT: float = 1.0  # side of the last-resort 1x1x1 marker


@dataclass
class PartialResults:
    """Best-effort intermediate outputs the answer heads publish as they firm up.

    Any field may be absent (None). The floor prefers a live head result when present
    and otherwise derives a legal answer straight from the scene.

    count:            deterministic set-cardinality if the numerical head has run.
    best_candidate:   top-ranked object-reference instance (pre-verification is fine).
    best_marker:      an explicit override marker (e.g. verified OR result).
    first_anchor_pt:  (x, y[, z]) of the best-grounded first IF sub-goal anchor.
    """

    count: int | None = None
    best_candidate: InstanceRecord | None = None
    best_marker: MarkerBox | None = None
    first_anchor_pt: Any | None = None


@dataclass
class _FloorCache:
    numerical: IntAnswer = field(default_factory=lambda: IntAnswer(MODAL_COUNT))
    object_reference: MarkerBox = field(
        default_factory=lambda: MarkerBox(0.0, 0.0, 0.0, _UNIT, _UNIT, _UNIT)
    )
    instruction_following: WaypointCmd = field(default_factory=lambda: WaypointCmd(0.0, 0.0))


def _target_noun(plan: Any) -> str | None:
    """Extract the target/first-anchor noun from a plan, tolerating any shape/None."""
    if plan is None:
        return None
    tgt = getattr(plan, "target", None)
    if tgt is not None:
        noun = getattr(tgt, "noun", None)
        if noun:
            return str(noun)
    route = getattr(plan, "route", None) or []
    for leg in route:
        for anchor in getattr(leg, "anchors", None) or []:
            noun = getattr(anchor, "noun", None)
            if noun:
                return str(noun)
    return None


def _anchor_nouns(plan: Any) -> list[str]:
    """Non-target reference nouns carried by the plan's target clauses, tolerating any
    shape/None (e.g. 'table' in 'find the teapot on the table' -> ['table']).

    Mirrors :func:`core.heads.explore_step._plan_nouns` but excludes the target noun
    itself — issue #42's anchor-guided rung wants only the ANCHOR side of a clause, not
    the target the rest of ``_object_reference`` is already trying (and failing) to
    match directly.
    """
    if plan is None:
        return []
    nouns: list[str] = []
    tgt = getattr(plan, "target", None)
    if tgt is not None:
        for clause in getattr(tgt, "clauses", None) or []:
            for anchor in getattr(clause, "anchors", None) or []:
                noun = getattr(anchor, "noun", None)
                if noun:
                    nouns.append(str(noun))
    return nouns


def _instances_for_noun(scene: SceneIndex | None, noun: str | None) -> list[InstanceRecord]:
    """Noun-matched instances via the scene's tolerant lookup; [] on any failure."""
    if scene is None or noun is None:
        return []
    try:
        return list(scene.by_label(noun))
    except Exception:
        return []


def _all_instances(scene: SceneIndex | None) -> list[InstanceRecord]:
    if scene is None:
        return []
    try:
        return list(scene.all_instances())
    except Exception:
        return []


def _xy(pt: Any) -> tuple[float, float] | None:
    """Coerce anything indexable/array-like to an (x, y) pair; None if impossible."""
    if pt is None:
        return None
    try:
        return float(pt[0]), float(pt[1])
    except Exception:
        return None


class FloorAnswers:
    """Holds the current floor answer for every qtype; refreshed each update()."""

    def __init__(self) -> None:
        self._cache = _FloorCache()

    # --------------------------------------------------------------- update
    def update(
        self,
        scene_index: SceneIndex | None,
        plan: Any | None,
        partial_results: PartialResults | None,
    ) -> None:
        """Recompute all three floor answers from the live scene. Never raises."""
        try:
            self._update_impl(scene_index, plan, partial_results)
        except Exception:
            # Last line of defence: keep whatever legal answer we already had.
            pass

    def _update_impl(
        self,
        scene: SceneIndex | None,
        plan: Any | None,
        partial: PartialResults | None,
    ) -> None:
        partial = partial or PartialResults()
        noun = _target_noun(plan)

        self._cache.numerical = self._numerical(scene, noun, partial)
        self._cache.object_reference = self._object_reference(scene, noun, partial, plan)
        self._cache.instruction_following = self._instruction_following(scene, partial)

    def _numerical(
        self, scene: SceneIndex | None, noun: str | None, partial: PartialResults
    ) -> IntAnswer:
        if partial.count is not None:
            return IntAnswer(int(partial.count))
        matches = _instances_for_noun(scene, noun)
        if matches:
            return IntAnswer(len(matches))
        return IntAnswer(MODAL_COUNT)

    def _object_reference(
        self,
        scene: SceneIndex | None,
        noun: str | None,
        partial: PartialResults,
        plan: Any | None = None,
    ) -> MarkerBox:
        # 1. explicit override / verified marker
        if partial.best_marker is not None:
            return partial.best_marker
        # 2. best-scored candidate handed down by the ranking head
        if partial.best_candidate is not None:
            return clamp_record_marker(partial.best_candidate)
        # 3. largest ANSWER-ELIGIBLE instance matching the target noun (issue #43a: a
        # target-noun instance under-observed (n_obs < 2) or too weakly scored
        # (peak score < 0.30) is a question-pass recall hit, not an answer — it stays in
        # the scene index (recall floor 0.25 untouched) but is skipped here so the floor
        # falls through to the anchor rung / final rung instead of marking a ghost.
        matches = [m for m in _instances_for_noun(scene, noun) if is_answer_eligible(m)]
        if matches:
            return clamp_record_marker(max(matches, key=_volume))
        # 4. anchor-guided (issue #42): the target noun is absent from the scene index,
        # but if the plan carries an anchor noun (e.g. 'table' in 'teapot on the table')
        # AND the index has instances for it, mark the best (largest) anchor instance
        # instead of falling straight to rung 5's blind largest-any-label guess — the
        # target usually sits ON/AT its anchor, so an anchor-centred marker is plausibly
        # within IoU distance of a small/undetected target, unlike an unrelated big box
        # elsewhere in the scene (the #42 repro: rung 5 alone landed on a column 1.4 m
        # from a mismarked teapot's GT). Anchors are tried in clause order; the first
        # anchor noun with any grounded instances wins.
        for anchor_noun in _anchor_nouns(plan):
            anchor_matches = _instances_for_noun(scene, anchor_noun)
            if anchor_matches:
                return clamp_record_marker(max(anchor_matches, key=_volume))
        # 5. any instance at all (largest, for a defensible box) — the final rung; silence
        # is still the only unforgivable failure, so this never withholds either.
        allinst = _all_instances(scene)
        if allinst:
            return clamp_record_marker(max(allinst, key=_volume))
        # 6. a 1x1x1 box at the most-observed cluster centroid (else origin)
        return self._unit_marker_at_most_observed(allinst)

    def _unit_marker_at_most_observed(self, allinst: list[InstanceRecord]) -> MarkerBox:
        if not allinst:
            return MarkerBox(0.0, 0.0, 0.0, _UNIT, _UNIT, _UNIT)
        best = max(allinst, key=lambda r: getattr(r, "n_obs", 0))
        c = best.centroid
        return MarkerBox(float(c[0]), float(c[1]), float(c[2]), _UNIT, _UNIT, _UNIT)

    def _instruction_following(
        self, scene: SceneIndex | None, partial: PartialResults
    ) -> WaypointCmd:
        # 1. best-grounded first anchor point handed down by the IF head
        xy = _xy(partial.first_anchor_pt)
        if xy is not None:
            return WaypointCmd(xy[0], xy[1])
        # 2. scene centroid (mean of instance centroids)
        allinst = _all_instances(scene)
        if allinst:
            pts = np.array([r.centroid[:2] for r in allinst], dtype=float)
            m = pts.mean(axis=0)
            return WaypointCmd(float(m[0]), float(m[1]))
        # 3. origin — never silence
        return WaypointCmd(0.0, 0.0)

    # --------------------------------------------------------------- get
    def get(self, qtype: QType) -> IntAnswer | MarkerBox | WaypointCmd:
        """Return the current publishable floor answer for qtype. Never raises."""
        if qtype is QType.NUMERICAL:
            return self._cache.numerical
        if qtype is QType.OBJECT_REFERENCE:
            return self._cache.object_reference
        return self._cache.instruction_following


def _volume(rec: InstanceRecord) -> float:
    """AABB volume of an instance; 0.0 on any degenerate/missing extent."""
    try:
        e = rec.extents
        return float(max(e[0], 0.0) * max(e[1], 0.0) * max(e[2], 0.0))
    except Exception:
        return 0.0
