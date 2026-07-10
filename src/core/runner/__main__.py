"""``python -m core.runner "How many chairs are near the table?"`` — drive one question.

Synthetic scene by default (deterministic, offline); ``--fixtures DIR`` replays a fixture
directory / bag instead. ``--verbose`` dumps the flight log. This is the single-question
half of the cockpit; the battery is ``python -m core.runner.battery``.
"""
from __future__ import annotations

import argparse
import sys

from core.interfaces import IntAnswer, MarkerBox, QType, WaypointCmd
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.parsing.regex_tier import classify_qtype
from core.runner.scenegen import build_scene_for
from core.runner.single import RunResult, run_question

_QTYPE_BUCKET = {
    QType.NUMERICAL: "numerical",
    QType.OBJECT_REFERENCE: "object_reference",
    QType.INSTRUCTION_FOLLOWING: "instruction_following",
}


def _synthetic_io(question_text: str, seed: int) -> MockRobotIO:
    """A synthetic scene containing an instance of every noun the question mentions.

    Route the question into its own qtype bucket so a numerical count target gets 2+
    instances (a count of 1 would be a degenerate scene).
    """
    bucket = _QTYPE_BUCKET[classify_qtype(question_text)]
    sc, _ = build_scene_for("_cli_", {bucket: [question_text]}, seed=seed)
    return MockRobotIO(sc, FakeClock(0.0))


def _replay_io(fixtures_dir: str):
    from core.replay.fixtures import load_fixtures
    from core.replay.replay_io import ReplayRobotIO

    store = load_fixtures(fixtures_dir)
    return ReplayRobotIO(store)


def _fmt_answer(ans) -> str:
    if isinstance(ans, IntAnswer):
        return f"IntAnswer(value={ans.value})"
    if isinstance(ans, MarkerBox):
        return f"MarkerBox(cx={ans.cx:.2f}, cy={ans.cy:.2f}, label={ans.label!r})"
    if isinstance(ans, WaypointCmd):
        return f"WaypointCmd(x={ans.x:.2f}, y={ans.y:.2f})"
    return repr(ans)


def _print_result(question: str, r: RunResult, *, verbose: bool) -> None:
    print(f"question : {question!r}")
    print(f"qtype    : {r.qtype.value if r.qtype else '?'}")
    print(f"answer   : {_fmt_answer(r.answer)}")
    print(f"published: {r.published}")
    print(f"floor    : {r.floor_used}")
    print(f"instances: {r.instances_tracked}")
    print(f"sim time : {r.elapsed_sim_s:.1f} s  ({r.ticks} ticks, wall {r.wall_s*1000:.0f} ms)")
    if r.end_of_data_at_sim_s is not None:
        print(f"eod at   : {r.end_of_data_at_sim_s:.1f} s (bag exhausted; frozen-world free-run after)")
    print(f"states   : {' -> '.join(r.states_visited)}")
    print(f"checkpt  : {r.checkpoint_calls} call(s)")
    if verbose:
        print("flight log:")
        for rec in r.flight_log:
            print(f"  [{rec.t:6.1f}s {rec.state:16s}] {rec.event}: {rec.detail}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="core.runner",
        description="Drive one question through the full pipeline and report.",
    )
    ap.add_argument("question", help="the challenge question text")
    ap.add_argument("--fixtures", default=None, help="replay a fixture directory / bag instead of synthetic")
    ap.add_argument(
        "--detections",
        default=None,
        help="scripted-labels JSON (jingfan_labels.json schema) to ground via the "
        "PerceptionPipeline; only valid with --fixtures. Real lidar fuses the scripted "
        "2D boxes into 3D instances the heads resolve against.",
    )
    ap.add_argument("--seed", type=int, default=0, help="synthetic-scene seed")
    ap.add_argument("--verbose", action="store_true", help="dump the flight log")
    ap.add_argument(
        "--tick-hz",
        type=float,
        default=1.0,
        help="simulated tick rate (default 1 Hz — coarser than the 5 Hz pipeline for "
        "cockpit responsiveness; phase gates are time-based so structure is unchanged)",
    )
    ap.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="scale the FSM budget gates (explore budgets, 510 s forced-assembly, 570 s "
        "watchdog) for replay against short bags; e.g. 0.2 puts the gates at ~102/114 s of "
        "bag time. Replay only (needs the ReplayRobotIO budget-clock seam).",
    )
    args = ap.parse_args(argv)

    if args.detections and not args.fixtures:
        ap.error("--detections requires --fixtures (it grounds against a replay's pano/scan stream)")

    if args.fixtures:
        io = _replay_io(args.fixtures)
    else:
        io = _synthetic_io(args.question, args.seed)

    result = run_question(
        args.question,
        io,
        detections_path=args.detections,
        tick_hz=args.tick_hz,
        budget_scale=args.budget_scale,
    )
    _print_result(args.question, result, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
