"""Run the 75 training questions against per-scene synthetic scenes; score STRUCTURAL health.

No ground-truth answers exist for the training set, so this is NOT an accuracy harness — it
is the whole-system health reading. Each question is driven end-to-end via
:func:`core.runner.single.run_question` over a synthetic scene built by
:func:`core.runner.scenegen.build_scene_for` (one instance per mentioned noun; 2+ for
numerical-count targets). Per run we record structural signals:

* answered_before_watchdog — a legal answer published before the 570 s floor
* answer_type_correct      — answer type matches the qtype
* parse_tier_used          — which parse rung produced the plan (offline => "regex")
* target_grounded          — OR/IF target/first-anchor noun resolves >= 1 scene candidate
* corridor_built / avoid_built (IF) — the plan carries a corridor leg / an avoid spec
* sim_elapsed              — simulated seconds at DONE
* floor_used               — the answer came from the watchdog/floor path, not a head

Outputs ``battery_report.md`` (per-scene table + aggregate rates) and
``battery_results.json`` into ``reports/battery_<YYYY-MM-DD>/`` (git-tracked history).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from core.interfaces import IntAnswer, MarkerBox, QType, WaypointCmd
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.parsing.regex_tier import parse_regex
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import LegKind
from core.plan_walk import iter_anchor_chain
from core.runner.scenegen import SceneSpec, build_scene_for
from core.runner.single import run_question

# ..\upstream\CMU-VLN-Challenge-2026\questions\questions.json relative to src/.
_SRC = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = _SRC.parent / "upstream" / "CMU-VLN-Challenge-2026" / "questions" / "questions.json"
DEFAULT_OUT_ROOT = _SRC.parent / "reports"

_WATCHDOG_FLOOR_S = 570.0

_TYPE_FOR_QTYPE = {
    QType.NUMERICAL: IntAnswer,
    QType.OBJECT_REFERENCE: MarkerBox,
    QType.INSTRUCTION_FOLLOWING: WaypointCmd,
}


@dataclass
class QuestionRun:
    """Structural health record for one driven question."""

    scene: str
    qtype: str
    question: str
    answered_before_watchdog: bool
    answer_type_correct: bool
    parse_tier_used: str
    target_grounded: bool
    corridor_built: bool
    avoid_built: bool
    sim_elapsed: float
    floor_used: bool
    checkpoint_calls: int
    wall_s: float
    plan_qtype: str = ""
    fsm_qtype: str = ""


def _floor_used(flight_log: list) -> bool:
    """True if the published answer came from the floor / watchdog path (not a head verify)."""
    for rec in flight_log:
        ev = getattr(rec, "event", "")
        if ev in ("answer_from_floor",):
            return True
        if ev == "published" and "watchdog" in getattr(rec, "detail", ""):
            return True
    return False


def _target_grounded(plan: Any, idx: BasicSceneIndex) -> bool:
    """OR/IF: does the target / first route-anchor noun resolve to >= 1 scene candidate?

    Route anchors recurse through nested ``disambiguator`` chains (issue #95): the
    "first route-anchor noun" is found by walking each anchor's chain in order, so a
    route anchor whose own noun is blank but whose disambiguator names something real
    is not silently treated as ungrounded.
    """
    tgt = getattr(plan, "target", None)
    if tgt is not None and getattr(tgt, "noun", None):
        return len(idx.by_label(tgt.noun)) >= 1
    route = getattr(plan, "route", None) or []
    for leg in route:
        for anchor in getattr(leg, "anchors", None) or []:
            for node in iter_anchor_chain(anchor):
                if getattr(node, "noun", None):
                    return len(idx.by_label(node.noun)) >= 1
    return False


def _legs(plan: Any) -> tuple[bool, bool]:
    """(corridor_built, avoid_built) for an instruction-following plan."""
    corridor = any(
        leg.kind is LegKind.CORRIDOR_BETWEEN for leg in (getattr(plan, "route", None) or [])
    )
    avoid = bool(getattr(plan, "avoid", None))
    return corridor, avoid


# Coarse tick rate for the battery: the FSM's phase gates are time-based (seconds off the
# sim clock), so a 1 Hz tick reproduces the same structural outcomes as 5 Hz while doing ~5x
# fewer (expensive) mock terrain/frontier recomputations — keeping each run well under a
# few wall-seconds. Correctness questions aren't scored here (no ground truth), only structure.
_BATTERY_TICK_HZ = 1.0


def _run_one(scene_name: str, qtype: QType, text: str, sc, idx: BasicSceneIndex) -> QuestionRun:
    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk)
    result = run_question(text, io, scene_index=idx, tick_hz=_BATTERY_TICK_HZ)

    plan = parse_regex(text)
    corridor, avoid = _legs(plan)
    # Score the answer type against the PLAN's qtype (authoritative), not the controller's
    # cheap pre-parse _infer_qtype heuristic — the two can diverge (e.g. "go near ... stop
    # at ..." infers OBJECT_REFERENCE but parses INSTRUCTION_FOLLOWING); the heads follow the
    # plan, so the plan is the correct expectation.
    want_type = _TYPE_FOR_QTYPE.get(plan.qtype)
    type_ok = result.answer is not None and want_type is not None and isinstance(
        result.answer, want_type
    )
    answered = result.published and result.elapsed_sim_s < _WATCHDOG_FLOOR_S

    grounded = qtype is QType.NUMERICAL or _target_grounded(plan, idx)

    return QuestionRun(
        scene=scene_name,
        qtype=qtype.value,
        question=text,
        answered_before_watchdog=answered,
        answer_type_correct=bool(type_ok),
        parse_tier_used=getattr(plan, "parse_tier", "?"),
        target_grounded=bool(grounded),
        corridor_built=corridor,
        avoid_built=avoid,
        sim_elapsed=round(result.elapsed_sim_s, 1),
        floor_used=_floor_used(result.flight_log),
        checkpoint_calls=result.checkpoint_calls,
        wall_s=round(result.wall_s, 4),
        plan_qtype=plan.qtype.value,
        fsm_qtype=result.qtype.value if result.qtype else "",
    )


_BUCKET_QTYPE = {
    "numerical": QType.NUMERICAL,
    "object_reference": QType.OBJECT_REFERENCE,
    "instruction_following": QType.INSTRUCTION_FOLLOWING,
}


def run_battery(
    questions_path: os.PathLike | str = DEFAULT_QUESTIONS,
    *,
    scenes: list[str] | None = None,
    seed: int = 0,
) -> tuple[list[QuestionRun], list[SceneSpec]]:
    """Drive every training question (optionally a scene subset) and collect structural runs."""
    with open(questions_path, encoding="utf-8") as fh:
        data = json.load(fh)

    runs: list[QuestionRun] = []
    specs: list[SceneSpec] = []
    for entry in data:
        scene_name = entry["scene"]
        if scenes is not None and scene_name not in scenes:
            continue
        questions = entry["questions"]
        sc, spec = build_scene_for(scene_name, questions, seed=seed)
        idx = BasicSceneIndex(sc.instances())
        specs.append(spec)
        for bucket, qtype in _BUCKET_QTYPE.items():
            for text in questions.get(bucket, []):
                runs.append(_run_one(scene_name, qtype, text, sc, idx))
    return runs, specs


# --------------------------------------------------------------------------- reporting


def _rate(runs: list[QuestionRun], attr: str) -> float:
    if not runs:
        return 0.0
    return sum(1 for r in runs if getattr(r, attr)) / len(runs)


def aggregate(runs: list[QuestionRun]) -> dict:
    """Aggregate structural rates overall and per qtype (report + json payload)."""
    def block(subset: list[QuestionRun]) -> dict:
        return {
            "n": len(subset),
            "answered_before_watchdog": round(_rate(subset, "answered_before_watchdog"), 3),
            "answer_type_correct": round(_rate(subset, "answer_type_correct"), 3),
            "target_grounded": round(_rate(subset, "target_grounded"), 3),
            "floor_used": round(_rate(subset, "floor_used"), 3),
        }

    per_type = {
        qt.value: block([r for r in runs if r.qtype == qt.value]) for qt in QType
    }
    if_runs = [r for r in runs if r.qtype == QType.INSTRUCTION_FOLLOWING.value]
    return {
        "overall": block(runs),
        "per_type": per_type,
        "corridor_built_rate_if": round(_rate(if_runs, "corridor_built"), 3),
        "avoid_built_rate_if": round(_rate(if_runs, "avoid_built"), 3),
        "parse_tiers": _tier_counts(runs),
    }


def _tier_counts(runs: list[QuestionRun]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in runs:
        out[r.parse_tier_used] = out.get(r.parse_tier_used, 0) + 1
    return out


def _md_table(runs: list[QuestionRun]) -> str:
    header = (
        "| Scene | QType | Answered<WD | Type OK | Grounded | Corridor | Avoid | Floor | Sim s | Q |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in runs:
        yn = lambda b: "yes" if b else "no"  # noqa: E731
        q = r.question if len(r.question) <= 60 else r.question[:57] + "..."
        rows.append(
            f"| {r.scene} | {r.qtype[:4]} | {yn(r.answered_before_watchdog)} | "
            f"{yn(r.answer_type_correct)} | {yn(r.target_grounded)} | "
            f"{yn(r.corridor_built)} | {yn(r.avoid_built)} | {yn(r.floor_used)} | "
            f"{r.sim_elapsed} | {q} |"
        )
    return header + "\n".join(rows) + "\n"


def write_report(
    runs: list[QuestionRun], specs: list[SceneSpec], out_dir: os.PathLike | str
) -> tuple[Path, Path]:
    """Write battery_report.md + battery_results.json into out_dir. Returns their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    agg = aggregate(runs)

    md_path = out / "battery_report.md"
    json_path = out / "battery_results.json"

    lines: list[str] = []
    lines.append(f"# Battery structural-health report ({date.today().isoformat()})\n")
    lines.append(
        f"{len(runs)} question(s) across {len(specs)} scene(s), regex-tier parse only "
        "(offline / deterministic). No ground-truth answers exist — these are STRUCTURAL "
        "health signals, not accuracy.\n"
    )
    o = agg["overall"]
    lines.append("## Aggregate\n")
    lines.append(
        f"- Answered before watchdog: **{o['answered_before_watchdog']:.0%}**\n"
        f"- Answer type correct: **{o['answer_type_correct']:.0%}**\n"
        f"- Target grounded (>=1 candidate): **{o['target_grounded']:.0%}**\n"
        f"- Floor/watchdog answer used: **{o['floor_used']:.0%}**\n"
        f"- IF corridor legs built: **{agg['corridor_built_rate_if']:.0%}**"
        f" | IF avoid specs built: **{agg['avoid_built_rate_if']:.0%}**\n"
        f"- Parse tiers: {agg['parse_tiers']}\n"
    )
    lines.append("### Per qtype\n")
    lines.append("| QType | n | Answered<WD | Type OK | Grounded | Floor |\n|---|---|---|---|---|---|\n")
    for qt, b in agg["per_type"].items():
        lines.append(
            f"| {qt} | {b['n']} | {b['answered_before_watchdog']:.0%} | "
            f"{b['answer_type_correct']:.0%} | {b['target_grounded']:.0%} | "
            f"{b['floor_used']:.0%} |"
        )
    lines.append("\n## Per-question\n")
    lines.append(_md_table(runs))

    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "date": date.today().isoformat(),
        "n_questions": len(runs),
        "n_scenes": len(specs),
        "aggregate": agg,
        "scene_specs": [asdict(s) for s in specs],
        "runs": [asdict(r) for r in runs],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return md_path, json_path


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: python -m core.runner.battery [--questions PATH] [--out DIR] [--scenes ...]."""
    ap = argparse.ArgumentParser(
        prog="core.runner.battery",
        description="Drive the training-question battery through the full pipeline (structural health).",
    )
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="questions.json path")
    ap.add_argument("--out", default=None, help="output dir (default reports/battery_<date>/)")
    ap.add_argument("--scenes", default=None, help="comma-separated scene subset")
    ap.add_argument("--seed", type=int, default=0, help="scene-synthesis seed")
    ap.add_argument(
        "--groundtruth",
        default=None,
        help="Unity root dir of GT scene folders; switches to battery v2 REAL accuracy "
        "scoring (delegates to core.runner.gt_battery) instead of structural health.",
    )
    args = ap.parse_args(argv)

    # Battery v2: REAL accuracy against ground-truth scenes.
    if args.groundtruth is not None:
        from core.runner import gt_battery

        gt_argv = ["--groundtruth", args.groundtruth, "--questions", args.questions]
        if args.out is not None:
            gt_argv += ["--out", args.out]
        if args.scenes is not None:
            gt_argv += ["--scenes", args.scenes]
        return gt_battery.main(gt_argv)

    scenes = [s.strip() for s in args.scenes.split(",")] if args.scenes else None
    out_dir = args.out or (DEFAULT_OUT_ROOT / f"battery_{date.today().isoformat()}")

    runs, specs = run_battery(args.questions, scenes=scenes, seed=args.seed)
    md_path, json_path = write_report(runs, specs, out_dir)

    agg = aggregate(runs)["overall"]
    print(
        f"battery: {len(runs)} questions / {len(specs)} scenes  "
        f"answered={agg['answered_before_watchdog']:.0%} "
        f"type_ok={agg['answer_type_correct']:.0%} "
        f"grounded={agg['target_grounded']:.0%} "
        f"floor={agg['floor_used']:.0%}"
    )
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
