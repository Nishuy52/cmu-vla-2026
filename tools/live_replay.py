"""LIVE-REPLAY harness (#163): turn one harvested cluster-verify run into an
offline, re-runnable regression test of TODAY's resolution/selection/counting
code, without a fresh 7.5 h cluster cycle.

Why this exists
----------------
Issue #163 quantified the gap between the offline ``gt_battery`` (a simulated
kinematic follower driven over GT-perfect indexes) and a real live run (a real
robot driven over live-PERCEIVED indexes): instruction_following rubric 0.889
offline vs 0.533 live, object_reference IoU 1.000 offline vs 0.000 live, 1
offline threading violation vs 5 live. The offline battery is not merely
optimistic — issue #163 shows it is sometimes anti-correlated with reality,
because it never runs the live perception/tracking stack that actually
produces a run's instances (duplicates, misses, noisy AABBs, ...).

A fresh cluster cycle to A/B a resolution/selection/counting fix costs ~7.5 h.
Most fixes in that family (leg-anchor tie-breaks, object_reference clause
verification, numerical counting gates, corridor-gate placement) do not change
WHAT got perceived — only what TODAY's code does with a FIXED set of already-
perceived instances. This module replays exactly that slice offline, in
seconds, from artifacts a cluster run already wrote to disk.

Two modes
---------
Mode 1 (resolution replay, ``python -m tools.live_replay resolve ...``):
    Rebuilds a :class:`~core.perception.scene_index.BasicSceneIndex` from the
    LAST record of a captured run's ``instance_index.jsonl`` (the live-
    perceived instance set at/near answer time — a frozen fact about that run,
    never regenerated) and the run's ``resolved_plan.jsonl`` (the ALREADY-
    PARSED :class:`~core.plan_schema.Plan` — parse-free, no LLM call), then
    re-runs TODAY's deterministic decision code against that frozen index:

    * object_reference -> :class:`core.heads.object_ref.ObjectRefHead`
      (``core.geometry.toolbox.resolve`` + the answer-eligibility gate).
    * numerical -> :class:`core.heads.numerical.NumericalHead`
      (``core.geometry.toolbox.counting`` + the H15(b) observation gate).
    * instruction_following -> per-leg anchor resolution via
      :meth:`core.heads.instruction.InstructionHead._resolve_leg_anchors`
      (the SAME private method the real head calls every tick) plus
      ``core.geometry.toolbox.corridor_gate`` for CORRIDOR_BETWEEN legs.

    Nothing here reimplements toolbox/head logic — it calls the exact same
    functions/classes ``src/core`` ships, so a change to that code changes
    this replay's answer identically to how it would change a live run's.

    "What the run answered" is read back from whatever artifact actually
    recorded it:

    * object_reference — the captured ``/selected_object_marker`` (read from
      the row's ``live.live_marker`` in a harvested ``scores.json``, or
      straight off the bag via ``tools.score_live_run.read_bag_capture`` if no
      scores row is available), matched to the nearest live-perceived
      instance by centroid distance. The marker and the instance-index
      positions share the SAME live/sim frame (the marker is literally
      ``clamp_record_marker()`` of a live ``InstanceRecord`` at publish time),
      so this match needs no GT frame fit.
    * numerical — the captured ``/numerical_response`` (same two sources).
    * instruction_following — there is no per-leg "what anchor id did the live
      run bind this leg to" artifact (the live dump surface only ever wrote
      relaxation/groundedness, never instance ids — see
      ``core.heads.instruction.dump_leg_relaxations``). The closest available
      fact is the LAST record of the run's ``relaxation_audit.jsonl`` (that
      same dump, at whatever tick the run's process last wrote it) — this
      tells us whether each leg/anchor was grounded and what relaxation rungs
      it took, which we diff against a fresh call to the same private method.
      It is a real, captured fact about the run, but it is NOT proof of what
      instance id got driven to — flagged explicitly in the output.

Mode 2 (rescore, ``python -m tools.live_replay rescore ...``):
    A thin pass-through to ``tools.score_live_run.main`` (that tool already
    does this; #163 says integrate, not duplicate).

What this CAN and CANNOT validate (read before trusting a "no change" result)
-------------------------------------------------------------------------------
CAN: any change confined to resolution/selection/counting/scoring logic over a
FIXED perceived-instance set — toolbox.resolve/counting/corridor_gate, the
object_reference/numerical head decision logic, leg anchor tie-breaks, scoring
formulas. This is exactly the class of change a battery-green/live-red result
from #163 showed the offline battery cannot be trusted to predict.

CANNOT: anything that would change WHICH instances got perceived, when, or how
confidently (perception/detection/tracking/NMS/association changes), or how
the robot explored/drove (exploration policy, planner, costmap, timing/budget
gates, the FSM's early-answer stability logic). The captured instance index
and driven trajectory are FROZEN inputs here — a fix in that territory needs a
real live run (or the offline battery, with its own known gap) to see any
effect at all. #163's numbers are exactly this: the object_reference
IoU 1.000-vs-0.000 and 1-vs-5 threading gaps were NOT resolution bugs (a
resolution replay of that run's own captured perception would still resolve
against the SAME imperfect instances a real run saw) — they trace to
perception/tracking/exploration, which only a live run (or a new offline
battery run, itself unable to reproduce them) can move.

Pure offline dev tool (tools/ — never part of the scored pipeline). CPU only.
No ROS/bag dependency for Mode 1 unless no ``scores.json`` row is available
for a run (falls back to reading the bag directly via ``rosbags``, already a
dependency of ``tools.score_live_run``).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import tools  # noqa: F401 -- inserts <repo>/src onto sys.path

from core.geometry import toolbox as TB
from core.geometry.toolbox import DEFAULT_THRESHOLDS, Thresholds
from core.heads.instruction import MIN_GROUND_OBS, InstructionHead
from core.heads.numerical import NumericalHead
from core.heads.object_ref import ObjectRefHead
from core.interfaces import ColorBin, InstanceRecord
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import LegKind, Plan, QType

_REPO = Path(__file__).resolve().parents[1]

#: Run-dir basename suffix -> QType value, matching the on-disk debug-slot naming
#: (``<idx>_<scene>_<qdir>``) used by ``tools.score_live_run``'s baseline layout AND the
#: ``reports/cluster_verify/<job>/debug/<slot>/`` harvest layout #163 points at.
QDIR_TO_QTYPE = {
    "nume": "numerical",
    "obje": "object_reference",
    "inst": "instruction_following",
}

# --------------------------------------------------------------------------- artifact IO


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def load_final_instance_snapshot(instance_index_path: Path) -> dict | None:
    """The LAST record of a run's ``instance_index.jsonl`` — the live-perceived
    instance set nearest to answer time (see module docstring)."""
    records = _read_jsonl(instance_index_path)
    return records[-1] if records else None


def load_resolved_plan(resolved_plan_path: Path) -> tuple[Plan, dict] | None:
    """The run's already-PARSED :class:`Plan` (parse-free — no LLM call), plus the
    raw record it came from. Prefers the last ``tag == "answer_time"`` record,
    falling back to the last record of any tag."""
    records = _read_jsonl(resolved_plan_path)
    if not records:
        return None
    answer_time = [r for r in records if r.get("tag") == "answer_time" and r.get("plan")]
    rec = answer_time[-1] if answer_time else records[-1]
    plan_dict = rec.get("plan")
    if plan_dict is None:
        return None
    return Plan.from_json(json.dumps(plan_dict)), rec


def load_last_relaxation_audit(relaxation_audit_path: Path) -> dict | None:
    """Last record of a run's ``relaxation_audit.jsonl`` (see module docstring's
    instruction_following caveat) — ``None`` if the run wrote no such file (only
    instruction_following runs do)."""
    records = _read_jsonl(relaxation_audit_path)
    return records[-1] if records else None


# --------------------------------------------------------------------------- snapshot -> index


def instance_record_from_dict(d: dict) -> InstanceRecord:
    """Reconstruct one :class:`InstanceRecord` from an
    ``instance_index.jsonl``/``dump_instance_index`` entry.

    Fields not carried by the dump (``points``, ``aliases``, OBB) are left at their
    dataclass defaults — none of Mode 1's decision code (resolve/counting/
    corridor_gate/answer-eligibility) reads them; every field they DO read
    (``instance_id``, ``label``, ``score``, ``n_obs``, ``centroid``, ``aabb_min``,
    ``aabb_max``, ``caption``, ``color_bins``) round-trips exactly.
    """
    color_bins = tuple(
        ColorBin(name=cb["name"], rgb=tuple(int(v) for v in cb["rgb"]), fraction=float(cb["fraction"]))
        for cb in d.get("color_bins", [])
    )
    return InstanceRecord(
        instance_id=int(d["id"]),
        label=str(d["label"]),
        score=float(d.get("score", 0.0)),
        n_obs=int(d.get("n_obs", 0)),
        centroid=np.asarray(d["position"], dtype=float),
        aabb_min=np.asarray(d["aabb_min"], dtype=float),
        aabb_max=np.asarray(d["aabb_max"], dtype=float),
        caption=str(d.get("caption", "")),
        color_bins=color_bins,
    )


def scene_index_from_snapshot(snapshot: dict) -> BasicSceneIndex:
    records = [instance_record_from_dict(d) for d in snapshot.get("instances", [])]
    return BasicSceneIndex(records)


# --------------------------------------------------------------------------- object_reference


@dataclass
class ObjectReferenceReplay:
    question: str
    then_marker: dict[str, float] | None
    then_matched_instance_id: int | None
    then_match_dist_m: float | None
    now_instance_id: int | None
    now_marker: dict[str, float] | None
    id_match: bool | None  # None when either side is unknown


def _nearest_instance_to_marker(
    marker: dict[str, float], index: BasicSceneIndex
) -> tuple[int | None, float | None]:
    """Nearest live-perceived instance (by centroid distance) to a marker's centre.

    Valid because the marker and the snapshot's instance positions share the SAME
    live/sim frame (module docstring) — no GT frame fit needed.
    """
    mx, my, mz = marker.get("cx"), marker.get("cy"), marker.get("cz")
    if mx is None:
        return None, None
    best_id: int | None = None
    best_d = math.inf
    for rec in index.all_instances():
        c = rec.centroid
        d = math.hypot(float(c[0]) - mx, float(c[1]) - my) if mz is None else math.sqrt(
            (float(c[0]) - mx) ** 2 + (float(c[1]) - my) ** 2 + (float(c[2]) - mz) ** 2
        )
        if d < best_d:
            best_d = d
            best_id = int(rec.instance_id)
    return best_id, (None if best_id is None else best_d)


def replay_object_reference(
    plan: Plan, index: BasicSceneIndex, then_marker: dict[str, float] | None,
    *, thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> ObjectReferenceReplay:
    head = ObjectRefHead(plan=plan, thresholds=thresholds)
    head.advance(index)
    now_marker_box = head.verify()
    now_id = None if head.best_candidate is None else int(head.best_candidate.instance_id)
    now_marker = None
    if now_marker_box is not None:
        now_marker = {
            "cx": now_marker_box.cx, "cy": now_marker_box.cy, "cz": now_marker_box.cz,
            "sx": now_marker_box.sx, "sy": now_marker_box.sy, "sz": now_marker_box.sz,
        }
    then_id: int | None = None
    then_dist: float | None = None
    if then_marker is not None:
        then_id, then_dist = _nearest_instance_to_marker(then_marker, index)
    id_match = None if (then_id is None or now_id is None) else (then_id == now_id)
    return ObjectReferenceReplay(
        question=plan.question_raw,
        then_marker=then_marker,
        then_matched_instance_id=then_id,
        then_match_dist_m=then_dist,
        now_instance_id=now_id,
        now_marker=now_marker,
        id_match=id_match,
    )


# --------------------------------------------------------------------------- numerical


@dataclass
class NumericalReplay:
    question: str
    then_answer: int | None
    now_count: int | None
    match: bool | None


def replay_numerical(
    plan: Plan, index: BasicSceneIndex, then_answer: int | None,
    *, thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> NumericalReplay:
    head = NumericalHead(plan=plan, thresholds=thresholds)
    head.advance(index)
    ans = head.answer()
    now_count = None if ans is None else int(ans.value)
    match = None if (then_answer is None or now_count is None) else (then_answer == now_count)
    return NumericalReplay(
        question=plan.question_raw, then_answer=then_answer, now_count=now_count, match=match,
    )


# --------------------------------------------------------------------------- instruction_following


@dataclass
class LegReplay:
    index: int
    kind: str
    nouns: tuple[str, ...]
    now_grounded: bool
    now_anchor_instance_ids: list[int | None]
    now_relax_steps: list[list[str]]
    now_goal_xy: tuple[float, float] | None
    now_gate: dict[str, Any] | None  # p0/p1/mid/width_m/degenerate, corridor legs only
    then_grounded: bool | None = None
    then_relax_steps: list[list[str]] | None = None
    grounded_changed: bool | None = None
    relax_steps_changed: bool | None = None


@dataclass
class InstructionReplay:
    question: str
    legs: list[LegReplay]
    then_source: str  # "relaxation_audit.jsonl" | "unavailable"


def _leg_relax_from_audit_record(audit_record: dict | None, leg_index: int) -> dict | None:
    if audit_record is None:
        return None
    for leg in audit_record.get("legs", []):
        if leg.get("index") == leg_index:
            return leg
    return None


def replay_instruction_following(
    plan: Plan, index: BasicSceneIndex, then_audit_record: dict | None,
    *, thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> InstructionReplay:
    head = InstructionHead(plan=plan, thresholds=thresholds)
    legs: list[LegReplay] = []
    prev_xy: tuple[float, float] | None = None
    for i, leg in enumerate(plan.route):
        recs, _provisional, audits = head._resolve_leg_anchors(leg, index, prev_xy)
        now_ids = [None if r is None else int(r.instance_id) for r in recs]
        now_steps = [list(a.steps) for a in audits]
        now_grounded = all(r is not None and r.n_obs >= MIN_GROUND_OBS for r in recs)

        now_goal_xy: tuple[float, float] | None = None
        now_gate: dict[str, Any] | None = None
        if leg.kind is LegKind.CORRIDOR_BETWEEN and recs[0] is not None and recs[1] is not None:
            gate = TB.corridor_gate(recs[0], recs[1], index)
            now_gate = {
                "p0": [float(gate.p0[0]), float(gate.p0[1])],
                "p1": [float(gate.p1[0]), float(gate.p1[1])],
                "mid": [float(gate.midpoint[0]), float(gate.midpoint[1])],
                "width_m": float(gate.width),
                "degenerate": bool(gate.degenerate),
            }
            prev_xy = (float(gate.midpoint[0]), float(gate.midpoint[1]))
        elif recs[0] is not None:
            c = recs[0].centroid
            now_goal_xy = (float(c[0]), float(c[1]))
            prev_xy = now_goal_xy
        # else: leg ungrounded -- prev_xy carried forward unchanged, mirroring
        # InstructionHead._ground_legs's own prev_xy handling.

        then_leg = _leg_relax_from_audit_record(then_audit_record, i)
        then_grounded = None if then_leg is None else bool(then_leg.get("grounded"))
        then_steps = None
        if then_leg is not None:
            then_steps = [list(a.get("relax_steps", [])) for a in then_leg.get("anchors", [])]
        grounded_changed = None if then_grounded is None else (then_grounded != now_grounded)
        relax_steps_changed = None if then_steps is None else (then_steps != now_steps)

        legs.append(
            LegReplay(
                index=i, kind=leg.kind.value, nouns=tuple(a.noun for a in leg.anchors),
                now_grounded=now_grounded, now_anchor_instance_ids=now_ids,
                now_relax_steps=now_steps, now_goal_xy=now_goal_xy, now_gate=now_gate,
                then_grounded=then_grounded, then_relax_steps=then_steps,
                grounded_changed=grounded_changed, relax_steps_changed=relax_steps_changed,
            )
        )
    return InstructionReplay(
        question=plan.question_raw, legs=legs,
        then_source="relaxation_audit.jsonl" if then_audit_record is not None else "unavailable",
    )


# --------------------------------------------------------------------------- run discovery


@dataclass
class RunRef:
    """One resolvable run: its debug dir, plus (scene, qdir, qtype) parsed from the
    debug-slot dirname and an optional matching ``scores.json`` row."""

    debug_dir: Path
    scene: str
    qdir: str
    qtype: str
    scores_row: dict | None = None


def _parse_slot_name(name: str) -> tuple[str, str] | None:
    """``"<idx>_<scene>_<qdir>"`` -> ``(scene, qdir)``, or None if unrecognised.

    ``idx`` is a bare integer prefix; ``qdir`` is the LAST underscore-delimited
    token and must be a known key of :data:`QDIR_TO_QTYPE` (scene names may
    themselves contain underscores, e.g. ``home_building_1``).
    """
    parts = name.split("_")
    if len(parts) < 3 or not parts[0].isdigit():
        return None
    qdir = parts[-1]
    if qdir not in QDIR_TO_QTYPE:
        return None
    scene = "_".join(parts[1:-1])
    return scene, qdir


def _load_scores_rows(job_dir: Path) -> list[dict]:
    scores_path = job_dir / "captures" / "scores.json"
    if not scores_path.is_file():
        return []
    try:
        data = json.loads(scores_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data.get("rows", [])


def _find_scores_row(rows: list[dict], scene: str, qdir: str) -> dict | None:
    for row in rows:
        if row.get("scene") == scene and row.get("qdir") == qdir:
            return row
    return None


def discover_runs(target: Path) -> list[RunRef]:
    """Resolve a CLI target to one or more :class:`RunRef`.

    ``target`` may be: a single debug slot (``.../debug/<slot>``, containing
    ``instance_index.jsonl`` directly), a job's ``debug/`` dir (all slots under
    it), or a job dir (``<job>/debug/*``).
    """
    if (target / "instance_index.jsonl").is_file():
        slots = [target]
        debug_root = target.parent
    elif target.name == "debug" and target.is_dir():
        slots = sorted(p for p in target.iterdir() if p.is_dir())
        debug_root = target
    elif (target / "debug").is_dir():
        slots = sorted(p for p in (target / "debug").iterdir() if p.is_dir())
        debug_root = target / "debug"
    else:
        return []
    job_dir = debug_root.parent
    rows = _load_scores_rows(job_dir)
    out: list[RunRef] = []
    for slot in slots:
        parsed = _parse_slot_name(slot.name)
        if parsed is None:
            continue
        scene, qdir = parsed
        out.append(
            RunRef(
                debug_dir=slot, scene=scene, qdir=qdir, qtype=QDIR_TO_QTYPE[qdir],
                scores_row=_find_scores_row(rows, scene, qdir),
            )
        )
    return out


# --------------------------------------------------------------------------- one-run replay


@dataclass
class RunReplayResult:
    scene: str
    qdir: str
    qtype: str
    debug_dir: str
    snapshot_n_instances: int | None
    snapshot_wall_time: float | None
    warnings: list[str] = field(default_factory=list)
    detail: dict[str, Any] | None = None


def replay_run(ref: RunRef, *, thresholds: Thresholds = DEFAULT_THRESHOLDS) -> RunReplayResult:
    warnings: list[str] = []
    snapshot = load_final_instance_snapshot(ref.debug_dir / "instance_index.jsonl")
    plan_rec = load_resolved_plan(ref.debug_dir / "resolved_plan.jsonl")

    if snapshot is None:
        warnings.append("no instance_index.jsonl records — cannot build a live-perceived index")
    if plan_rec is None:
        warnings.append("no resolved_plan.jsonl records — cannot re-run resolution")
    if snapshot is None or plan_rec is None:
        return RunReplayResult(
            scene=ref.scene, qdir=ref.qdir, qtype=ref.qtype, debug_dir=str(ref.debug_dir),
            snapshot_n_instances=None, snapshot_wall_time=None, warnings=warnings, detail=None,
        )

    plan, _plan_raw = plan_rec
    index = scene_index_from_snapshot(snapshot)

    detail: dict[str, Any]
    if plan.qtype is QType.OBJECT_REFERENCE:
        then_marker = None
        if ref.scores_row is not None:
            then_marker = ref.scores_row.get("live", {}).get("live_marker")
        if then_marker is None:
            warnings.append("no scores.json row / live_marker — 'answered-then' side unknown")
        detail = asdict(replay_object_reference(plan, index, then_marker, thresholds=thresholds))
    elif plan.qtype is QType.NUMERICAL:
        then_answer = None
        if ref.scores_row is not None:
            then_answer = ref.scores_row.get("live", {}).get("live_answer")
        if then_answer is None:
            warnings.append("no scores.json row / live_answer — 'answered-then' side unknown")
        detail = asdict(replay_numerical(plan, index, then_answer, thresholds=thresholds))
    elif plan.qtype is QType.INSTRUCTION_FOLLOWING:
        then_audit = load_last_relaxation_audit(ref.debug_dir / "relaxation_audit.jsonl")
        if then_audit is None:
            warnings.append(
                "no relaxation_audit.jsonl — 'answered-then' groundedness/relax-steps unknown"
            )
        result = replay_instruction_following(plan, index, then_audit, thresholds=thresholds)
        detail = {"question": result.question, "then_source": result.then_source,
                   "legs": [asdict(leg) for leg in result.legs]}
    else:
        warnings.append(f"unhandled qtype {plan.qtype!r}")
        detail = {}

    return RunReplayResult(
        scene=ref.scene, qdir=ref.qdir, qtype=ref.qtype, debug_dir=str(ref.debug_dir),
        snapshot_n_instances=int(snapshot.get("total_instances", len(snapshot.get("instances", [])))),
        snapshot_wall_time=snapshot.get("wall_time"), warnings=warnings, detail=detail,
    )


# --------------------------------------------------------------------------- reporting


def _fmt_bool_delta(then: bool | None, now: bool, changed: bool | None) -> str:
    if then is None:
        return f"now={now} (then unknown)"
    marker = " CHANGED" if changed else ""
    return f"then={then} now={now}{marker}"


def build_markdown(results: list[RunReplayResult]) -> str:
    lines = ["# live-replay: resolution replay (mode 1)", ""]
    n_changed = 0
    n_total = 0
    for r in results:
        lines.append(f"## {r.scene} / {r.qdir} ({r.qtype})")
        lines.append(f"- debug dir: `{r.debug_dir}`")
        lines.append(
            f"- snapshot: {r.snapshot_n_instances} instances @ wall_time={r.snapshot_wall_time}"
        )
        for w in r.warnings:
            lines.append(f"- WARNING: {w}")
        d = r.detail or {}
        if r.qtype == "object_reference" and d:
            n_total += 1
            changed = d.get("id_match") is False
            n_changed += int(changed)
            lines.append(f"- question: {d.get('question')!r}")
            lines.append(
                f"- then-matched instance id: {d.get('then_matched_instance_id')} "
                f"(dist={d.get('then_match_dist_m')})"
            )
            lines.append(f"- now-resolved instance id: {d.get('now_instance_id')}")
            lines.append(f"- id_match: {d.get('id_match')}" + (" CHANGED" if changed else ""))
        elif r.qtype == "numerical" and d:
            n_total += 1
            changed = d.get("match") is False
            n_changed += int(changed)
            lines.append(f"- question: {d.get('question')!r}")
            lines.append(f"- then answer: {d.get('then_answer')}  now count: {d.get('now_count')}")
            lines.append(f"- match: {d.get('match')}" + (" CHANGED" if changed else ""))
        elif r.qtype == "instruction_following" and d:
            lines.append(f"- question: {d.get('question')!r}")
            lines.append(f"- then-source: {d.get('then_source')}")
            for leg in d.get("legs", []):
                n_total += 1
                changed = bool(leg.get("grounded_changed") or leg.get("relax_steps_changed"))
                n_changed += int(changed)
                lines.append(
                    f"  - leg {leg['index']} ({leg['kind']}, nouns={leg['nouns']}): "
                    + _fmt_bool_delta(leg.get("then_grounded"), leg["now_grounded"], leg.get("grounded_changed"))
                    + f"; now_anchor_ids={leg['now_anchor_instance_ids']}"
                )
                if leg.get("relax_steps_changed"):
                    lines.append(
                        f"    relax_steps then={leg.get('then_relax_steps')} now={leg.get('now_relax_steps')}"
                    )
        lines.append("")
    lines.insert(1, f"\n{n_changed}/{n_total} comparisons changed vs the run's own record.\n")
    return "\n".join(lines)


def build_json(results: list[RunReplayResult]) -> dict:
    return {"n_runs": len(results), "runs": [asdict(r) for r in results]}


# --------------------------------------------------------------------------- CLI


def _cmd_resolve(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="tools.live_replay resolve")
    parser.add_argument(
        "targets", nargs="+",
        help="debug slot dir(s), job debug/ dir(s), or job dir(s) (auto-discovers slots)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write <out>.json + <out>.md")
    args = parser.parse_args(argv)

    refs: list[RunRef] = []
    for t in args.targets:
        refs.extend(discover_runs(Path(t)))
    if not refs:
        print("no runs discovered under: " + ", ".join(args.targets), file=sys.stderr)
        return 1

    results = [replay_run(ref) for ref in refs]
    md = build_markdown(results)
    js = build_json(results)
    print(md)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".md").write_text(md, encoding="utf-8")
        args.out.with_suffix(".json").write_text(json.dumps(js, indent=2), encoding="utf-8")
        print(f"wrote {args.out.with_suffix('.md')} + {args.out.with_suffix('.json')}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("resolve", "rescore"):
        print(__doc__.split("\n\n")[0])
        print("\nusage: python -m tools.live_replay {resolve,rescore} ...", file=sys.stderr)
        return 2
    mode, rest = argv[0], argv[1:]
    if mode == "rescore":
        from tools import score_live_run  # Mode 2 -- integrate, never duplicate (#163)
        return score_live_run.main(rest)
    return _cmd_resolve(rest)


if __name__ == "__main__":
    raise SystemExit(main())
