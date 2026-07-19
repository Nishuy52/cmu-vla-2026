"""Leave-3-scenes-out cross-validated calibration of the reasoning thresholds.

This is the Phase-2 sweep harness. It searches the high-sensitivity reasoning
calibration parameters (the geometry :class:`~core.geometry.toolbox.Thresholds`
family) against the ground-truth battery objective, under 5-fold cross-validation
(5 folds x 3 held-out scenes over the 15 training scenes), and reports a robust
per-parameter recommendation that generalises across folds rather than overfitting one
scene split.

Why CV, not a single fit
------------------------
A single global argmax over all 15 scenes overfits: a threshold that happens to flip
one borderline predicate in one scene wins with no evidence it helps elsewhere. Leaving
3 scenes out per fold and freezing the fold's best config on the held-out scenes
measures *generalisation* — the gap between train and held-out score is the honest read
on whether a value transfers. The final recommendation keeps only values that recur
across folds (modal consensus); parameters with no consensus are flagged
``unstable — keep default`` rather than moved on thin evidence.

The objective (composite, mirrors the challenge points) — repaired (NUM-F7 / H2)
--------------------------------------------------------------------------------
Per scene-set, per config, we score exactly what the *repaired* scorers now emit — the
retired artifact metrics (planned-path coverage for IF; class-only-inclusive count
agreement) are gone:

* **NUMERICAL (1 pt each):** STRICT independent count agreement — our count equals the
  ``referential`` independent second opinion, over questions that HAVE strict independent
  evidence only. Relation-agnostic ``*_class_only`` rows are NOT evidence (NUM-F6/F7:
  comparing a class-total against a relation-filtered count is a scorer artifact) and are
  EXCLUDED from both numerator and denominator, exactly as ``gt_battery.aggregate`` now
  does (``independent_source == "referential"``). Scene-graph counts are diagnostics, not
  the objective target.
* **OBJECT_REFERENCE (2 pts each):** IoU >= 0.25 against the GT target, over the
  *scoreable* questions only (a GT target was matched, honestly, via the vocab bridge /
  referential / uniqueness). Unscoreable ("none"/"ambiguous") questions are EXCLUDED.
* **INSTRUCTION_FOLLOWING (6 pts each):** the DRIVEN-trajectory RUBRIC score
  (``score_instruction_rubric`` — ordered per-leg arrival credit minus threading/avoid
  penalties over the *driven* trajectory, IF-F2), over the *aligned* questions only
  (scene frame fit residual <= 1 m). This replaces the old planned-path coverage@1m,
  which the challenge rubric does not pay for. Unaligned questions are EXCLUDED.

The scene-set score is ``sum(points earned) / sum(points available)`` over the included
questions, so a scene-set with more IF weight is not penalised for having fewer OR
questions. Exclusion counts are reported alongside every score.

Threshold injection
-------------------
The sweep moves the geometry ``Thresholds`` (all wired via function params in
``toolbox.py``) — but ONLY into the pipeline/subject under test, never into the
measuring instrument. Concretely: swept ``Thresholds`` reach ``score_numerical`` /
``score_object_reference`` (which use them solely to compute OUR resolver's answer —
the GT target/independent count those answers are checked against is threshold-
independent text/annotation matching) and the ``InstructionHead`` that drives the IF
trajectory (:func:`_drive_if_trajectory`). The scoring.py functions are imported and
reused verbatim — never forked; only the gt_battery *orchestration* (scene mirror,
drive loop) is mirrored here so the swept thresholds can be threaded through the
subject. The rubric's REFERENCE geometry — leg goals, corridor gates, avoid capsules
(:func:`_if_rubric_geometry`) and the terminal-goal scene-alignment anchor
(:func:`_terminal_goal_centroid`) — is the measuring instrument, not the subject: it is
resolved with the FIXED default thresholds, identically for every candidate config, by
delegating directly to gt_battery's own (default-threshold) copies of those two
helpers. This closes an instrument-endogeneity hole: letting the rubric geometry or the
alignment gate move with the swept config would let a candidate score higher by
shrinking the rubric denominator (de-resolving legs/gates) or de-aligning a
zero-scoring scene out of ``if_available``, rather than by driving better. Budget/nav/
fusion params that are hard module constants (no injection seam today — see
docs/calibration.md "Phase-2 wiring TODO") are NOT in the default spec; if a requested
key cannot be injected it is reported and skipped, never hacked in.

``counting.min_obs`` was DROPPED from the sweep space (NUM-F7): every GT instance has
``n_obs=3`` so the gate is provably inert on GT data — it was dead weight, never a
tunable that could move the objective.

Runtime & determinism
---------------------
gt_battery per scene is fast (fully-observed, no exploration), but IF trajectory driving
dominates (~2-3 s/scene). We therefore cache each ``(config-hash, scene)`` evaluation:
the same config is scored once per scene and the result reused across every fold that
holds that scene in or out. The cache is BOTH in-memory (per run) and ON DISK (a JSON
file per ``(config-hash, scene)`` under ``<out>/cache/``), so an interrupted sweep
resumes cheaply — a re-run skips every cell already computed instead of recomputing it.
All randomness (fold assignment, random search) is seeded, so a given
``(seed, spec, scenes)`` reproduces bit-for-bit. ``--n-samples`` and ``--scenes-subset``
bound the work for smoke runs.

Pure/offline: numpy + stdlib only, no network, no new dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from core.calibration import (
    Calibration,
    apply_overrides,
    default_calibration,
    diff,
)
from core.groundtruth import scoring as S
from core.groundtruth.loader import GTScene, load_scene
from core.interfaces import QType
from core.runner import gt_battery as GB
from core.runner.provenance import collect_provenance

_SRC = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = GB.DEFAULT_QUESTIONS
DEFAULT_QUESTIONS_ROOT = GB.DEFAULT_QUESTIONS_ROOT
DEFAULT_OUT_ROOT = _SRC.parent / "reports"

# Point weights mirroring the challenge scoring (points-weighted objective).
_PTS_NUMERICAL = 1.0
_PTS_OBJECT_REFERENCE = 2.0
_PTS_INSTRUCTION_FOLLOWING = 6.0

_OR_IOU_HIT = 0.25  # IoU >= this counts as a correct object reference
# Instrument constant (frozen, never swept): scenes above this fit residual are excluded
# (unaligned). Single-sourced from scoring.py so the gate can't drift from gt_battery's.
_IF_ALIGN_RESIDUAL_GATE_M = S._ALIGN_RESIDUAL_GATE_M


# --------------------------------------------------------------------------- sweep spec


@dataclass(frozen=True)
class SweepSpec:
    """A search space: dotted calibration keys -> candidate values.

    ``params`` maps a dotted ``subsystem.field`` key (as consumed by
    :func:`core.calibration.apply_overrides`) to the list of candidate values to try for
    it. The random search draws one value per key per sample. A key must be *wireable*
    (reach its consumer via a function param / constructor arg) — the default spec only
    lists geometry + counting params, which are; unwireable keys are rejected up front by
    :meth:`validate` so a typo or a module-constant param fails loudly instead of silently
    no-oping.
    """

    params: dict[str, list[Any]]

    def keys(self) -> list[str]:
        return sorted(self.params)

    def validate(self) -> None:
        """Reject unknown keys (apply_overrides raises) and non-empty candidate lists.

        Every swept key must be a live ``Calibration`` field (checked via
        ``apply_overrides``, which raises ``KeyError`` on an unknown subsystem/field). This
        is the grid-vs-ledger consistency guard: if a future rename drops or renames a
        Thresholds field that the grid still references, ``validate`` fails loudly here
        (and in ``tests/runner/test_cvsweep_objective.py``) instead of silently no-oping.
        """
        cal = default_calibration()
        for key, values in self.params.items():
            if not values:
                raise ValueError(f"sweep key {key!r} has no candidate values")
            # apply_overrides raises KeyError on an unknown subsystem/field.
            apply_overrides(cal, {key: values[0]})

    def sample(self, rng: random.Random) -> dict[str, Any]:
        """Draw one override dict: one candidate value per key (seeded)."""
        return {k: rng.choice(self.params[k]) for k in self.keys()}

    def grid_size(self) -> int:
        n = 1
        for v in self.params.values():
            n *= len(v)
        return n


def default_sweep_spec() -> SweepSpec:
    """High-sensitivity reasoning params from docs/calibration.md, 3-5 values each.

    Chosen for the answer-path predicates the challenge questions actually exercise:
    the ``near``/``on``/``above``/``next_to``/``in`` geometry family (H/M sensitivity).
    Every key is a live geometry ``Thresholds`` field, wireable today via the toolbox
    function params, and matches the CURRENT field set (H5 renamed/added
    ``on_upper_span_frac`` / ``on_top_tol`` / ``above_lateral_infl`` and retired the old
    ``on_vert_tol``). Budget/nav/fusion module-constant params are deliberately excluded
    (see docstring / calibration.md wiring TODO).

    The numerical ``counting.min_obs`` gate was DROPPED (NUM-F7): it is provably inert on
    GT data (every instance has ``n_obs=3``), so it could never move the objective —
    including it was dead weight, not a tunable.
    """
    return SweepSpec(
        params={
            # near radius = max(near_floor, near_scale * footprint_diag)
            "geometry.near_floor": [0.8, 1.0, 1.2, 1.5, 2.0],
            "geometry.near_scale": [0.4, 0.5, 0.6, 0.75, 0.9],
            # adjacency / support. on() is the H5 support-semantics form: IoM-over-min
            # gate + upper z-span band (the old top-face-only on_vert_tol is retired).
            # above() uses the H5 lateral-offset inflation (footprint-overlap gate
            # replaced) so wall-hung objects over a headboard qualify.
            "geometry.next_to_gap": [0.5, 0.75, 1.0, 1.25],
            "geometry.on_min_overlap_frac": [0.30, 0.45, 0.50, 0.60],
            "geometry.on_upper_span_frac": [0.0, 0.15, 0.25, 0.40],
            "geometry.on_top_tol": [0.10, 0.15, 0.20, 0.30],
            "geometry.above_lateral_infl": [0.25, 0.50, 0.75, 1.0],
            # containment
            "geometry.in_containment_frac": [0.45, 0.60, 0.75],
        }
    )


# --------------------------------------------------------------------------- config id


def config_hash(overrides: dict[str, Any]) -> str:
    """Stable short hash of an override dict (order-independent, value-typed).

    Used as the cache key component and the human-facing config id in the report. Two
    override dicts that differ only in key order hash identically; an int vs float value
    difference does not (``1`` vs ``1.0`` are distinct configs).
    """
    payload = json.dumps(
        {k: [type(overrides[k]).__name__, overrides[k]] for k in sorted(overrides)},
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- scene scoring


@dataclass
class SceneQScores:
    """Per-question challenge outcomes for one scene under one config.

    Only the fields the objective needs, already reduced to points-earned/available so
    the fold aggregation is a simple sum. Exclusion counts are carried for reporting.
    """

    numerical_earned: float = 0.0
    numerical_available: float = 0.0
    numerical_excluded: int = 0
    or_earned: float = 0.0
    or_available: float = 0.0
    or_excluded: int = 0
    if_earned: float = 0.0
    if_available: float = 0.0
    if_excluded: int = 0

    def add(self, other: "SceneQScores") -> None:
        for f in (
            "numerical_earned", "numerical_available", "numerical_excluded",
            "or_earned", "or_available", "or_excluded",
            "if_earned", "if_available", "if_excluded",
        ):
            setattr(self, f, getattr(self, f) + getattr(other, f))

    @property
    def earned(self) -> float:
        return self.numerical_earned + self.or_earned + self.if_earned

    @property
    def available(self) -> float:
        return self.numerical_available + self.or_available + self.if_available

    def composite(self) -> float:
        """Points earned / points available (0.0 when nothing is scoreable)."""
        return self.earned / self.available if self.available > 0 else 0.0


def _strict_independent(rec: GB.GTQuestionScore) -> int | None:
    """The STRICT independent count for a numerical question, or None (NUM-F6/F7).

    Only a ``referential`` (relation-aware) independent count is evidence — the retired
    behaviour of falling back to scene-graph and accepting ``*_class_only`` relation-
    agnostic totals inflated/artifact-poisoned the agreement stat (F6/F7). This mirrors
    ``gt_battery.aggregate``'s ``independent_source == "referential"`` filter exactly:
    a question is included in the numerical objective only when it carries strict
    referential evidence; otherwise it is excluded (returns None), never scored against a
    class-only or scene-graph number.
    """
    if rec.independent_source == "referential" and rec.gt_count_independent is not None:
        return rec.gt_count_independent
    return None


def score_scene(
    gt: GTScene,
    questions: dict[str, list[str]],
    *,
    referential: dict | None,
    scene_graph: dict | None,
    questions_dir: os.PathLike | str | None,
    thresholds,
) -> SceneQScores:
    """Threshold-aware challenge scoring of one scene (mirrors gt_battery.score_scene).

    Reuses gt_battery's per-question records + scoring.py's scorers, threading the swept
    ``thresholds`` into the numerical/OR scorers and the IF trajectory driver, then
    reduces each question to points-earned/available under the *repaired* points-weighted
    objective (rubric IF x6, strict count x1, IoU OR x2; unscoreable / unaligned /
    no-strict-evidence questions excluded). gt_battery itself is untouched.
    """
    from core.perception.scene_index import BasicSceneIndex

    idx = BasicSceneIndex(gt.instances)
    out = SceneQScores()

    # --- numerical: STRICT independent count agreement (1 pt) ------------------------
    # min_obs stays at the scorer default (1): NUM-F7 dropped it from the sweep because
    # it is provably inert on GT data (every instance has n_obs=3), so there is nothing
    # to thread — the scorer's own default is faithful.
    for text in questions.get("numerical", []):
        ns = S.score_numerical(
            text, idx, referential=referential, scene_graph=scene_graph,
            thresholds=thresholds,
        )
        rec = GB.GTQuestionScore(
            scene=gt.scene_name, qtype=QType.NUMERICAL.value, question=text,
            our_count=ns.our_count, gt_count_independent=ns.gt_count_independent,
            independent_source=ns.independent_source,
            gt_count_scenegraph=ns.gt_count_scenegraph,
        )
        indep = _strict_independent(rec)
        if indep is None:
            out.numerical_excluded += 1
            continue
        out.numerical_available += _PTS_NUMERICAL
        if ns.our_count == indep:
            out.numerical_earned += _PTS_NUMERICAL

    # --- object reference: IoU >= 0.25 on scoreable questions (2 pts) ----------------
    for text in questions.get("object_reference", []):
        ors = S.score_object_reference(
            text, idx, gt.instances, referential=referential, thresholds=thresholds,
        )
        if ors.iou != ors.iou:  # NaN -> unscoreable (no honest GT target)
            out.or_excluded += 1
            continue
        out.or_available += _PTS_OBJECT_REFERENCE
        if ors.iou >= _OR_IOU_HIT:
            out.or_earned += _PTS_OBJECT_REFERENCE

    # --- instruction following: driven-trajectory RUBRIC on aligned questions (6 pts) -
    if_texts = questions.get("instruction_following", [])
    if if_texts:
        out.add(
            _score_if(
                if_texts, gt, idx, questions_dir=questions_dir, thresholds=thresholds,
            )
        )
    return out


def _score_if(
    if_texts: list[str],
    gt: GTScene,
    idx,
    *,
    questions_dir: os.PathLike | str | None,
    thresholds,
) -> SceneQScores:
    """IF scoring for a scene: fit the scene frame, drive each trajectory, rubric-score it.

    Retargeted to the IF-F2 HEADLINE metric: resolve each terminal goal + load the GT
    trajectory, fit one scene transform, then for each aligned question DRIVE the
    trajectory (constant-speed kinematic follower over the planned breadcrumbs) under
    the swept ``thresholds`` (the SUBJECT under test) and score it with
    :func:`core.groundtruth.scoring.score_instruction_rubric` against reference geometry
    resolved at FIXED default thresholds (the INSTRUMENT — see module docstring
    "Threshold injection"; delegates to gt_battery's ``_terminal_goal_centroid`` /
    ``_if_rubric_geometry``) — ordered per-leg arrival credit minus threading/avoid
    penalties. The planned-path coverage@1m the sweep used to optimise is retired (the
    rubric does not pay for path shape). Questions whose scene fit residual exceeds the
    alignment gate (or that lack a GT trajectory) are excluded; the alignment gate is
    computed from the fixed-threshold goal, so it too cannot be gamed by ``thresholds``.
    """
    out = SceneQScores()
    if_traj: list[np.ndarray | None] = []
    if_goal: list[np.ndarray | None] = []
    for i, text in enumerate(if_texts):
        traj_q = GB._IF_TRAJ_INDEX.get(i)
        traj_arr: np.ndarray | None = None
        if questions_dir is not None and traj_q is not None:
            cand = Path(questions_dir) / gt.scene_name / f"trajectory_q{traj_q}.ply"
            if cand.exists():
                traj_arr = S.load_trajectory_ply(cand)
        if_traj.append(traj_arr)
        # INSTRUMENT (frozen): the scene-alignment goal is reference geometry, not the
        # system under test — resolve it with gt_battery's fixed-default helper so the
        # alignment gate below can't be gamed by the swept `thresholds` (see module
        # docstring "Threshold injection").
        if_goal.append(
            GB._terminal_goal_centroid(text, idx) if traj_arr is not None else None
        )

    pairs = [(t, g) for t, g in zip(if_traj, if_goal) if t is not None and t.shape[0] > 0]
    frame, residual = S.align_scene_trajectories(pairs) if pairs else (None, None)

    spawn_xy: tuple[float, float] | None = None
    if frame is not None and pairs:
        start_pt = pairs[0][0][0, :2]
        mapped = frame.apply(np.asarray([start_pt], dtype=float))[0]
        spawn_xy = (float(mapped[0]), float(mapped[1]))

    aligned = (
        frame is not None
        and residual is not None
        and residual <= _IF_ALIGN_RESIDUAL_GATE_M
    )

    for i, text in enumerate(if_texts):
        traj_q = GB._IF_TRAJ_INDEX.get(i)
        traj_path = None
        if questions_dir is not None and traj_q is not None:
            cand = Path(questions_dir) / gt.scene_name / f"trajectory_q{traj_q}.ply"
            if cand.exists():
                traj_path = cand
        if traj_path is None or not aligned:
            out.if_excluded += 1
            continue
        # SUBJECT (swept): the pipeline under test drives with the candidate thresholds.
        driven = _drive_if_trajectory(text, gt, idx, thresholds, start_xy=spawn_xy)
        # INSTRUMENT (frozen): the rubric's reference geometry (leg goals / gates /
        # avoid capsules) must be identical for every candidate config, or a config can
        # shrink the rubric denominator by de-resolving anchors instead of driving
        # better (see module docstring "Threshold injection"). Delegate to gt_battery's
        # copy, which hard-wires the default Thresholds.
        leg_goals, corridor_gates, avoid_caps, _leg_iids = GB._if_rubric_geometry(text, gt, idx)
        rub = S.score_instruction_rubric(
            driven,
            leg_goals,
            corridor_gates=corridor_gates,
            avoid_capsules=avoid_caps,
            trajectory_ply=traj_path,
            frame=frame,
        )
        out.if_available += _PTS_INSTRUCTION_FOLLOWING
        out.if_earned += _PTS_INSTRUCTION_FOLLOWING * float(rub.rubric_score)
    return out


def _drive_if_trajectory(text, gt, idx, thresholds, *, start_xy=None) -> np.ndarray:
    """Threshold-aware mirror of gt_battery._drive_if_trajectory (IF-F2 driven poses).

    Builds + grounds the route under the swept ``thresholds`` (InstructionHead(thresholds=)),
    then closes the loop: repeatedly asks the follower for the next crumb and advances a
    constant-speed kinematic vehicle toward it, recording each pose. This driven pose
    stream — not the planned path — is what the rubric scores. Mirrors gt_battery's copy
    (which hard-wires the default thresholds); the drive-loop constants are imported from
    gt_battery so the two stay in lock-step.
    """
    from core.parsing.regex_tier import parse_regex
    from core.heads.instruction import InstructionHead
    from core.mocks.mock_io import FakeClock, MockRobotIO

    plan = parse_regex(text)
    if plan.qtype is not QType.INSTRUCTION_FOLLOWING or not plan.route:
        return np.empty((0, 2), dtype=float)

    sc = GB._synthetic_from_gt(gt)
    clk = FakeClock(0.0)
    if start_xy is not None:
        start_x, start_y = float(start_xy[0]), float(start_xy[1])
    else:
        start_x = float(min(r.aabb_min[0] for r in gt.instances)) + 0.5
        start_y = float(min(r.aabb_min[1] for r in gt.instances)) + 0.5
    io = MockRobotIO(sc, clk, start_x=start_x, start_y=start_y)

    head = InstructionHead(plan=plan, thresholds=thresholds)
    for _ in range(GB._IF_MAX_BUILD_TICKS):
        head.advance(io, idx)
        clk.advance(1.0)
        if head._follower is not None and head._follower.path:
            break

    follower = head._follower
    if follower is None or not follower.path:
        return np.empty((0, 2), dtype=float)

    odom = io.latest_odom()
    pose = (float(odom.x), float(odom.y)) if odom is not None else (0.0, 0.0)
    t = 0.0
    poses: list[tuple[float, float]] = [pose]
    for _ in range(GB._DRIVE_MAX_TICKS):
        wp = follower.advance(pose, t)
        if wp is None:
            break
        target = (float(wp.x), float(wp.y))
        dx, dy = target[0] - pose[0], target[1] - pose[1]
        d = (dx * dx + dy * dy) ** 0.5
        if d <= GB._DRIVE_STEP_M:
            pose = target
        else:
            pose = (pose[0] + GB._DRIVE_STEP_M * dx / d, pose[1] + GB._DRIVE_STEP_M * dy / d)
        poses.append(pose)
        t += 1.0
    term = follower.path[-1]
    if not poses or (poses[-1][0] - term[0]) ** 2 + (poses[-1][1] - term[1]) ** 2 > (
        GB._DRIVE_STEP_M**2
    ):
        poses.append((float(term[0]), float(term[1])))
    return np.array(poses, dtype=float)


def _drive_if_path(text, gt, idx, thresholds, *, start_xy=None) -> np.ndarray:
    """Retained diagnostic helper: the PLANNED breadcrumb path (not scored by the objective).

    The objective now scores the DRIVEN trajectory via :func:`_drive_if_trajectory`; this
    planned-path mirror is kept only for ad-hoc diagnostics/tests. Threshold-aware copy of
    gt_battery._drive_if_path.
    """
    from core.parsing.regex_tier import parse_regex
    from core.heads.instruction import InstructionHead
    from core.mocks.mock_io import FakeClock, MockRobotIO

    plan = parse_regex(text)
    if plan.qtype is not QType.INSTRUCTION_FOLLOWING or not plan.route:
        return np.empty((0, 2), dtype=float)

    sc = GB._synthetic_from_gt(gt)
    clk = FakeClock(0.0)
    if start_xy is not None:
        start_x, start_y = float(start_xy[0]), float(start_xy[1])
    else:
        start_x = float(min(r.aabb_min[0] for r in gt.instances)) + 0.5
        start_y = float(min(r.aabb_min[1] for r in gt.instances)) + 0.5
    io = MockRobotIO(sc, clk, start_x=start_x, start_y=start_y)

    head = InstructionHead(plan=plan, thresholds=thresholds)
    for _ in range(GB._IF_MAX_BUILD_TICKS):
        head.advance(io, idx)
        clk.advance(1.0)
        if head._follower is not None and head._follower.path:
            break

    follower = head._follower
    if follower is None or not follower.path:
        if io.waypoints:
            return np.array([[w.x, w.y] for w in io.waypoints], dtype=float)
        return np.empty((0, 2), dtype=float)
    return np.array([[p[0], p[1]] for p in follower.path], dtype=float)


# --------------------------------------------------------------------------- folds


def make_folds(
    scenes: list[str], *, n_folds: int = 5, holdout_size: int = 3, seed: int = 0
) -> list[list[str]]:
    """Deterministic partition of scenes into ``n_folds`` disjoint held-out groups.

    The scene names are sorted (stable input), shuffled once under ``seed``, then sliced
    into ``n_folds`` contiguous groups of ``holdout_size``. With 15 scenes / 5 folds / 3
    each this is an exact partition (every scene held out exactly once). Returns the list
    of held-out groups; the train set for fold *k* is every scene not in group *k*.
    """
    if n_folds * holdout_size != len(scenes):
        raise ValueError(
            f"{len(scenes)} scenes cannot split into {n_folds} folds x {holdout_size} "
            "(need n_folds*holdout_size == n_scenes)"
        )
    ordered = sorted(scenes)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return [ordered[i * holdout_size : (i + 1) * holdout_size] for i in range(n_folds)]


# --------------------------------------------------------------------------- cache / eval


#: The flat numeric/int fields of :class:`SceneQScores`, in a fixed order — the on-disk
#: cache payload schema. Kept as a module constant so serialise/deserialise agree and a
#: field add/remove is a one-line change with a loud round-trip test.
_SCENEQ_FIELDS = (
    "numerical_earned", "numerical_available", "numerical_excluded",
    "or_earned", "or_available", "or_excluded",
    "if_earned", "if_available", "if_excluded",
)


def _sceneq_to_dict(s: SceneQScores) -> dict[str, float]:
    return {f: getattr(s, f) for f in _SCENEQ_FIELDS}


def _sceneq_from_dict(d: dict[str, Any]) -> SceneQScores:
    out = SceneQScores()
    for f in _SCENEQ_FIELDS:
        # excluded counts are ints; earned/available are floats. Coerce off the default's
        # type so a JSON round-trip stays type-stable.
        cur = getattr(out, f)
        val = d[f]
        setattr(out, f, int(val) if isinstance(cur, int) else float(val))
    return out


class SceneEvaluator:
    """Evaluates ``(config, scene)`` -> :class:`SceneQScores`, cached across folds.

    Loads each scene's GT once (lazily, cached) and each ``(config-hash, scene)`` score
    once. Folds that share a scene in/out reuse the same evaluation — the expensive IF
    trajectory driving runs a single time per unique config+scene pair.

    Two cache tiers:

    * **in-memory** (``_eval_cache``): reuse within a single run across folds.
    * **on-disk** (``cache_dir``, one JSON file per ``(config-hash, scene)``): survives an
      interrupted process, so a re-run resumes — every cell already on disk is a hit and is
      not recomputed. Enabled when a ``cache_dir`` is passed (the CLI points it at
      ``<out>/cache/``); ``None`` keeps the harness memory-only (used by fast unit tests).
    """

    def __init__(
        self,
        unity_root: os.PathLike | str,
        questions_by_scene: dict[str, dict[str, list[str]]],
        *,
        questions_dir: os.PathLike | str | None,
        base_cal: Calibration | None = None,
        cache_dir: os.PathLike | str | None = None,
    ) -> None:
        self.root = Path(unity_root)
        self.questions_by_scene = questions_by_scene
        self.questions_dir = questions_dir
        self.base_cal = base_cal or default_calibration()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._scene_cache: dict[str, tuple] = {}
        self._eval_cache: dict[tuple[str, str], SceneQScores] = {}
        self.n_cache_hits = 0
        self.n_cache_misses = 0
        self.n_disk_hits = 0  # subset of misses served from a prior run's on-disk cell

    def _load(self, scene: str):
        if scene not in self._scene_cache:
            folder = GB._find_scene_folder(self.root, scene)
            if folder is None:
                raise FileNotFoundError(f"scene folder not found for {scene!r} under {self.root}")
            gt = load_scene(folder, scene_name=scene)
            referential = GB._load_referential(folder, scene)
            scene_graph = GB._load_scene_graph(folder, scene)
            self._scene_cache[scene] = (gt, referential, scene_graph)
        return self._scene_cache[scene]

    def _disk_path(self, cfg_hash: str, scene: str) -> Path | None:
        if self.cache_dir is None:
            return None
        # scene names are challenge identifiers (safe filename tokens); cfg_hash is hex.
        return self.cache_dir / f"{cfg_hash}__{scene}.json"

    def _disk_load(self, cfg_hash: str, scene: str) -> SceneQScores | None:
        p = self._disk_path(cfg_hash, scene)
        if p is None or not p.exists():
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            return _sceneq_from_dict(payload["scores"])
        except (json.JSONDecodeError, KeyError, OSError):
            # A truncated/corrupt cell (e.g. killed mid-write) is treated as absent and
            # recomputed — never a crash that aborts the resume.
            return None

    def _disk_store(self, cfg_hash: str, scene: str, res: SceneQScores) -> None:
        p = self._disk_path(cfg_hash, scene)
        if p is None:
            return
        payload = {"config_hash": cfg_hash, "scene": scene, "scores": _sceneq_to_dict(res)}
        # Atomic-ish write: tmp then replace, so an interrupt can't leave a half file that
        # the resume would mistake for a valid cell.
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, p)

    def evaluate(self, overrides: dict[str, Any], scene: str) -> SceneQScores:
        cfg_hash = config_hash(overrides)
        key = (cfg_hash, scene)
        if key in self._eval_cache:
            self.n_cache_hits += 1
            return self._eval_cache[key]
        # On-disk resume: a cell computed by a prior (interrupted) run is a hit — load it
        # into memory and skip recomputation.
        disk = self._disk_load(cfg_hash, scene)
        if disk is not None:
            self.n_cache_hits += 1
            self.n_disk_hits += 1
            self._eval_cache[key] = disk
            return disk
        self.n_cache_misses += 1
        gt, referential, scene_graph = self._load(scene)
        cal = apply_overrides(self.base_cal, overrides)
        res = score_scene(
            gt,
            self.questions_by_scene[scene],
            referential=referential,
            scene_graph=scene_graph,
            questions_dir=self.questions_dir,
            thresholds=cal.geometry,
        )
        self._eval_cache[key] = res
        self._disk_store(cfg_hash, scene, res)
        return res

    def score_scene_set(self, overrides: dict[str, Any], scenes: Iterable[str]) -> SceneQScores:
        agg = SceneQScores()
        for sc in scenes:
            agg.add(self.evaluate(overrides, sc))
        return agg


# --------------------------------------------------------------------------- CV sweep


@dataclass
class FoldResult:
    fold: int
    holdout: list[str]
    train_score: float
    holdout_score: float
    best_overrides: dict[str, Any]
    best_config_hash: str


@dataclass
class SweepResult:
    spec_keys: list[str]
    n_samples: int
    seed: int
    scenes: list[str]
    folds: list[FoldResult]
    stability: dict[str, dict[str, int]]  # key -> {value_repr: count across folds}
    recommendation: dict[str, Any]  # key -> chosen value (modal) or "keep default"
    recommended_overrides: dict[str, Any]  # only the keys that moved off default
    cache_hits: int
    cache_misses: int
    disk_hits: int = 0  # cache hits served from a prior run's on-disk cell (resume)

    def mean_holdout(self) -> float:
        return float(np.mean([f.holdout_score for f in self.folds])) if self.folds else 0.0

    def std_holdout(self) -> float:
        return float(np.std([f.holdout_score for f in self.folds])) if self.folds else 0.0

    def mean_train(self) -> float:
        return float(np.mean([f.train_score for f in self.folds])) if self.folds else 0.0

    def generalization_gap(self) -> float:
        return self.mean_train() - self.mean_holdout()


def _value_repr(v: Any) -> str:
    """Stable string key for a candidate value in the stability table."""
    return repr(v)


def run_cv_sweep(
    evaluator: SceneEvaluator,
    spec: SweepSpec,
    folds: list[list[str]],
    *,
    n_samples: int = 60,
    seed: int = 0,
) -> SweepResult:
    """Random-search each fold on its train scenes, freeze on held-out; aggregate.

    Draws ``n_samples`` seeded configs from ``spec`` (the SAME sample set for every fold,
    so folds differ only by scene split, not by which configs they saw — a fair
    comparison). The default (all-baseline) config is always included so a fold can pick
    'no change' when nothing beats it. For each fold: pick the config with the highest
    train-scene composite, evaluate it frozen on the held-out scenes. Then build the
    per-parameter stability table (how often each chosen value recurs) and the modal
    recommendation.
    """
    spec.validate()
    rng = random.Random(seed)
    samples: list[dict[str, Any]] = [{}]  # {} == baseline default (no overrides)
    seen = {config_hash({})}
    for _ in range(n_samples):
        cfg = spec.sample(rng)
        h = config_hash(cfg)
        if h not in seen:
            seen.add(h)
            samples.append(cfg)

    fold_results: list[FoldResult] = []
    for k, holdout in enumerate(folds):
        train = [s for s in evaluator.questions_by_scene if s not in holdout]
        best_cfg: dict[str, Any] = {}
        best_train = -1.0
        for cfg in samples:
            sc = evaluator.score_scene_set(cfg, train).composite()
            # Deterministic tie-break: prefer the earlier (fewer-change) config; samples
            # is ordered with baseline first, so ties keep the simpler config.
            if sc > best_train:
                best_train = sc
                best_cfg = cfg
        holdout_score = evaluator.score_scene_set(best_cfg, holdout).composite()
        fold_results.append(
            FoldResult(
                fold=k,
                holdout=holdout,
                train_score=round(best_train, 6),
                holdout_score=round(holdout_score, 6),
                best_overrides=dict(best_cfg),
                best_config_hash=config_hash(best_cfg),
            )
        )

    stability, recommendation, rec_over = _aggregate_stability(spec, fold_results)
    return SweepResult(
        spec_keys=spec.keys(),
        n_samples=n_samples,
        seed=seed,
        scenes=sorted(evaluator.questions_by_scene),
        folds=fold_results,
        stability=stability,
        recommendation=recommendation,
        recommended_overrides=rec_over,
        cache_hits=evaluator.n_cache_hits,
        cache_misses=evaluator.n_cache_misses,
        disk_hits=evaluator.n_disk_hits,
    )


def _aggregate_stability(
    spec: SweepSpec, folds: list[FoldResult]
) -> tuple[dict[str, dict[str, int]], dict[str, Any], dict[str, Any]]:
    """Per-parameter stability table + modal recommendation.

    For each swept key, count how often each chosen value appears across folds (a fold
    that did not move the key off default contributes its *default* value). A value that
    is the unique mode AND recurs in a strict majority of folds becomes the recommendation;
    otherwise the key is flagged ``unstable — keep default`` and left at its default.
    ``recommended_overrides`` holds only the keys whose modal recommendation differs from
    the default (the diff to apply).
    """
    cal = default_calibration()
    defaults: dict[str, Any] = {}
    for key in spec.keys():
        sub, _, field_name = key.partition(".")
        defaults[key] = getattr(getattr(cal, sub), field_name)

    stability: dict[str, dict[str, int]] = {}
    recommendation: dict[str, Any] = {}
    rec_over: dict[str, Any] = {}
    n_folds = len(folds)
    for key in spec.keys():
        counter: Counter[str] = Counter()
        value_by_repr: dict[str, Any] = {}
        for fr in folds:
            val = fr.best_overrides.get(key, defaults[key])
            r = _value_repr(val)
            counter[r] += 1
            value_by_repr[r] = val
        stability[key] = dict(counter)
        # modal value + strict-majority consensus gate
        top_r, top_n = counter.most_common(1)[0]
        modes = [r for r, n in counter.items() if n == top_n]
        if len(modes) == 1 and top_n > n_folds / 2:
            chosen = value_by_repr[top_r]
            recommendation[key] = chosen
            if _value_repr(chosen) != _value_repr(defaults[key]):
                rec_over[key] = chosen
        else:
            recommendation[key] = "unstable — keep default"
    return stability, recommendation, rec_over


# --------------------------------------------------------------------------- report


def _load_questions(
    questions_path: os.PathLike | str, scenes: list[str] | None
) -> dict[str, dict[str, list[str]]]:
    """questions.json -> {scene: {qtype: [texts]}} restricted to ``scenes`` if given."""
    with open(questions_path, encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[str, dict[str, list[str]]] = {}
    for entry in data:
        scene = entry["scene"]
        if scenes is not None and scene not in scenes:
            continue
        out[scene] = entry["questions"]
    return out


def write_report(
    result: SweepResult,
    out_dir: os.PathLike | str,
    *,
    argv: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    """Write report.md, results.json, recommended_calibration.json; return their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "report.md"
    json_path = out / "results.json"
    rec_path = out / "recommended_calibration.json"

    cal = default_calibration()
    # Every recommended override is a live geometry Calibration field (counting.min_obs was
    # dropped from the sweep space, NUM-F7), so they apply directly.
    recommended_cal = apply_overrides(cal, result.recommended_overrides)
    rec_diff = diff(cal, recommended_cal)

    # --- report.md -------------------------------------------------------------------
    L: list[str] = []
    L.append(f"# CV calibration sweep — reasoning thresholds ({date.today().isoformat()})\n")
    L.append(
        f"5-fold leave-3-out CV over {len(result.scenes)} scenes, {result.n_samples} "
        f"random-search samples (seed {result.seed}). Cache: {result.cache_hits} hits "
        f"({result.disk_hits} from a prior on-disk run — resume) / "
        f"{result.cache_misses} misses.\n"
    )
    L.append(
        "> **Objective (repaired, NUM-F7/H2):** points-weighted challenge composite — "
        "NUMERICAL STRICT independent (referential) count agreement (1pt; class-only / "
        "scene-graph rows excluded, not evidence), OBJECT_REFERENCE IoU>=0.25 on scoreable "
        "Qs (2pt), INSTRUCTION_FOLLOWING driven-trajectory RUBRIC score on aligned Qs (6pt; "
        "the retired planned-path coverage@1m is NOT scored). Normalised by available "
        "points; unscoreable / unaligned / no-strict-evidence questions excluded from both "
        "numerator and denominator.\n"
    )
    L.append("## Fold scores\n")
    L.append("| Fold | Held-out scenes | Train | Holdout | Best config (diff vs default) |")
    L.append("|---|---|---|---|---|")
    for fr in result.folds:
        d = fr.best_overrides
        diff_str = ", ".join(f"{k}={v}" for k, v in sorted(d.items())) or "(default)"
        L.append(
            f"| {fr.fold} | {', '.join(fr.holdout)} | {fr.train_score:.3f} | "
            f"{fr.holdout_score:.3f} | {diff_str} |"
        )
    L.append("")
    L.append(
        f"- **Mean holdout:** {result.mean_holdout():.3f} +/- {result.std_holdout():.3f}\n"
        f"- **Mean train:** {result.mean_train():.3f}\n"
        f"- **Generalization gap (train - holdout):** {result.generalization_gap():.3f}\n"
    )

    L.append("## Per-parameter stability (chosen value counts across folds)\n")
    L.append("| Param | Default | Value counts across folds | Recommendation |")
    L.append("|---|---|---|---|")
    for key in result.spec_keys:
        default_v = getattr(getattr(cal, key.split(".")[0]), key.split(".")[1])
        counts = ", ".join(
            f"{v}×{n}" for v, n in sorted(result.stability[key].items(), key=lambda kv: -kv[1])
        )
        rec = result.recommendation[key]
        L.append(f"| {key} | {default_v} | {counts} | {rec} |")
    L.append("")

    def _default_of(key: str) -> Any:
        sub, _, fld = key.partition(".")
        return getattr(getattr(cal, sub), fld)

    L.append("## Final recommendation\n")
    if result.recommended_overrides:
        L.append("Robust (modal, strict-majority) parameter moves off default:\n")
        for k in sorted(result.recommended_overrides):
            L.append(f"- `{k}`: {_default_of(k)} -> {result.recommended_overrides[k]}")
        L.append("")
    else:
        L.append(
            "No parameter reached strict-majority modal consensus across folds — every "
            "swept key is left at its default (the sweep found no value that robustly "
            "generalises; the defaults are the honest recommendation).\n"
        )
    L.append(
        "Parameters flagged `unstable — keep default` had no single value recur in a "
        "strict majority of folds — moving them would overfit one scene split.\n"
    )
    md_path.write_text("\n".join(L), encoding="utf-8")

    # --- results.json ----------------------------------------------------------------
    payload = {
        "date": date.today().isoformat(),
        # The swept subject varies thresholds per-config, but the sweep *instrument* and
        # its reported identity are the default calibration — stamp that (meth-F7/F8).
        "provenance": collect_provenance("cvsweep", argv, cal),
        "n_samples": result.n_samples,
        "seed": result.seed,
        "scenes": result.scenes,
        "spec_keys": result.spec_keys,
        "mean_holdout": round(result.mean_holdout(), 6),
        "std_holdout": round(result.std_holdout(), 6),
        "mean_train": round(result.mean_train(), 6),
        "generalization_gap": round(result.generalization_gap(), 6),
        "cache_hits": result.cache_hits,
        "cache_misses": result.cache_misses,
        "disk_hits": result.disk_hits,
        "folds": [
            {
                "fold": fr.fold,
                "holdout": fr.holdout,
                "train_score": fr.train_score,
                "holdout_score": fr.holdout_score,
                "best_overrides": fr.best_overrides,
                "best_config_hash": fr.best_config_hash,
            }
            for fr in result.folds
        ],
        "stability": result.stability,
        "recommendation": result.recommendation,
        "recommended_overrides": result.recommended_overrides,
        "recommended_diff_vs_default": rec_diff,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- recommended_calibration.json ------------------------------------------------
    from core.calibration import to_json
    rec_path.write_text(to_json(recommended_cal), encoding="utf-8")
    return md_path, json_path, rec_path


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="core.runner.cvsweep",
        description="Leave-3-scenes-out CV calibration of the reasoning thresholds.",
    )
    ap.add_argument(
        "--groundtruth", required=True,
        help="Unity root dir with per-scene folders (matched to questions.json scenes).",
    )
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="questions.json path")
    ap.add_argument(
        "--questions-dir", default=str(DEFAULT_QUESTIONS_ROOT),
        help="dir holding <scene>/trajectory_q*.ply",
    )
    ap.add_argument("--out", default=None, help="output dir (default reports/cvsweep_<date>/)")
    ap.add_argument("--n-samples", type=int, default=60, help="random-search sample count")
    ap.add_argument("--seed", type=int, default=0, help="seed for folds + random search")
    ap.add_argument(
        "--scenes-subset", default=None,
        help="comma-separated scene subset (smoke runs); folds adapt to the count",
    )
    ap.add_argument(
        "--n-folds", type=int, default=5, help="number of CV folds (default 5)"
    )
    ap.add_argument(
        "--holdout-size", type=int, default=3, help="held-out scenes per fold (default 3)"
    )
    ap.add_argument(
        "--no-cache", action="store_true",
        help="disable the on-disk (config,scene) cache under <out>/cache/ (no resume)",
    )
    args = ap.parse_args(argv)

    scenes_subset = (
        [s.strip() for s in args.scenes_subset.split(",")] if args.scenes_subset else None
    )
    questions_by_scene = _load_questions(args.questions, scenes_subset)
    scenes = sorted(questions_by_scene)
    if not scenes:
        print("cvsweep: no scenes selected")
        return 1

    n_folds, holdout = _fit_folds(len(scenes), args.n_folds, args.holdout_size)
    folds = make_folds(scenes, n_folds=n_folds, holdout_size=holdout, seed=args.seed)

    # Resolve the output dir up front so the on-disk cache lives under it (<out>/cache/):
    # an interrupted run resumes from the same --out on re-invocation.
    out_dir = Path(args.out) if args.out else (DEFAULT_OUT_ROOT / f"cvsweep_{date.today().isoformat()}")
    cache_dir = None if args.no_cache else out_dir / "cache"

    evaluator = SceneEvaluator(
        args.groundtruth, questions_by_scene, questions_dir=args.questions_dir,
        cache_dir=cache_dir,
    )
    spec = default_sweep_spec()
    result = run_cv_sweep(evaluator, spec, folds, n_samples=args.n_samples, seed=args.seed)

    stamp_argv = list(argv) if argv is not None else sys.argv[1:]
    md_path, json_path, rec_path = write_report(result, out_dir, argv=stamp_argv)

    print(
        f"cvsweep: {len(scenes)} scenes / {n_folds} folds x {holdout}  "
        f"mean_holdout={result.mean_holdout():.3f}+/-{result.std_holdout():.3f} "
        f"gap={result.generalization_gap():.3f} "
        f"cache={result.cache_hits}h({result.disk_hits}disk)/{result.cache_misses}m"
    )
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    print(f"wrote {rec_path}")
    return 0


def _fit_folds(n_scenes: int, want_folds: int, want_holdout: int) -> tuple[int, int]:
    """Adapt fold geometry to the scene count for smoke subsets.

    Prefers the requested ``(want_folds, want_holdout)`` when it exactly partitions
    ``n_scenes``; otherwise falls back to holdout=1 with n_folds=n_scenes (leave-one-out),
    which always partitions. This keeps ``--scenes-subset`` runs valid without the caller
    computing a compatible split.
    """
    if want_folds * want_holdout == n_scenes:
        return want_folds, want_holdout
    return n_scenes, 1


if __name__ == "__main__":
    sys.exit(main())
