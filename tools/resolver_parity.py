"""Resolve-outcome parity audit between the rubric's goal resolution
(``core.runner.gt_battery._if_rubric_geometry``) and the navigation head's
own anchor resolution (``core.heads.instruction.InstructionHead``).

Issue #71: both sides call ``core.geometry.toolbox.resolve()`` against the
SAME ground-truth scene index and are meant to agree on which GT instance a
route leg's anchor refers to. This tool runs BOTH resolution paths on
identical scene inputs for every IF question's every leg, logs each side's
resolved instance id + class + goal xy per leg, and tables any leg where the
two sides land on a different instance or a goal-position delta above
``--flag-dist-m``.

Deliberately kept OUT of the scored path (``core.runner.gt_battery`` is not
modified to run this by default) -- offline diagnostic only, same convention
as ``tools/battery_diff.py``.

Usage (from the repo root)::

    python -m tools.resolver_parity --groundtruth <unity_root> [--out <path.json>]

Read-only: drives the real ``_run_instruction_head`` / ``_if_rubric_geometry``
call sites unmodified so this reports genuine production behaviour, not a
reimplementation of either side's logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.groundtruth.loader import load_scene  # noqa: E402
from core.interfaces import QType  # noqa: E402
from core.parsing.regex_tier import parse_regex  # noqa: E402
from core.perception.scene_index import BasicSceneIndex  # noqa: E402
from core.plan_schema import LegKind  # noqa: E402
from core.runner import gt_battery as GB  # noqa: E402

_FLAG_DIST_M_DEFAULT = 0.5  # any nonzero divergence is interesting; default flags nearly all


def _rubric_side(text: str, gt, idx, start_xy):
    """Call the REAL ``_if_rubric_geometry`` (production, unmodified) and re-pad its
    output back to one slot per ``plan.route`` entry -- that function silently
    ``continue``s past a leg whose anchor doesn't resolve, so its returned lists can be
    SHORTER than ``plan.route``; we re-walk the route re-resolving only to detect which
    positions were skipped (a leg with no anchors, or every anchor failing to resolve),
    never to recompute the geometry -- the goal/id values themselves always come
    straight from ``_if_rubric_geometry``'s own return.
    """
    plan = parse_regex(text)
    rows: list[dict] = []
    if not plan.route:
        return rows

    leg_goals, _gates, _avoids, leg_ids, _aabbs = GB._if_rubric_geometry(
        text, gt, idx, start_xy=start_xy
    )

    # _if_rubric_geometry skips (not pads) legs whose anchors don't resolve, so its
    # output can be shorter than plan.route. Re-derive which route positions were
    # skipped using ONLY resolvability (never re-deriving geometry) so the padded rows
    # line up positionally with plan.route / the head's ``_legs``.
    from core.geometry.toolbox import TargetSpec, resolve as _resolve

    def _resolvable(anchor) -> bool:
        spec = TargetSpec(
            noun=anchor.noun, raw=anchor.raw, attributes=list(anchor.attributes),
            clauses=[anchor.disambiguator] if anchor.disambiguator is not None else [],
        )
        return bool(_resolve(spec, idx).candidates_ranked)

    gi = 0  # index into the (shorter-or-equal) _if_rubric_geometry output
    for leg in plan.route:
        skipped = not leg.anchors or not all(_resolvable(a) for a in leg.anchors)
        if skipped or gi >= len(leg_goals):
            kind = (
                "corridor_between" if leg.kind is LegKind.CORRIDOR_BETWEEN
                else ("via_near" if leg.kind is LegKind.VIA_NEAR else "goto")
            )
            rows.append({"kind": kind, "ids": None, "goal": None})
            continue
        kind, goal = leg_goals[gi]
        ids = leg_ids[gi] if gi < len(leg_ids) else None
        rows.append({"kind": kind, "ids": tuple(ids) if ids else None, "goal": goal})
        gi += 1
    return rows


def _head_side(text: str, gt, idx, start_xy, wall_cells):
    """Run the real ``InstructionHead`` (via the production
    ``_run_instruction_head``, unmodified) and read back each leg's resolved
    anchor record(s) + geometry straight off ``head._legs``.
    """
    head, _io, plan = GB._run_instruction_head(
        text, gt, idx, start_xy=start_xy, wall_cells=wall_cells,
        max_build_ticks=GB._IF_MAX_BUILD_TICKS,
    )
    rows: list[dict] = []
    if head is None or not plan.route:
        return rows
    for i, leg in enumerate(plan.route):
        gl = head._legs[i] if i < len(head._legs) else None
        if gl is None or gl.record is None:
            rows.append({"kind": leg.kind.value, "ids": None, "goal": None})
            continue
        if leg.kind is LegKind.CORRIDOR_BETWEEN:
            # _GroundedLeg only retains the FIRST anchor's record (see
            # core.heads.instruction._ground_one) -- re-resolve the second
            # anchor the same way _resolve_leg_anchors does (DISTINCT from the
            # first) so both corridor endpoints are comparable. Pure/no side
            # effects: mirrors production _resolve_anchor_distinct exactly.
            prev_xy = None
            if i > 0 and head._legs[i - 1].geom is not None:
                g = head._legs[i - 1].geom
                prev_xy = g[1] if isinstance(g[0], tuple) else g
            used = {gl.record.instance_id}
            r1, _prov = head._resolve_anchor_distinct(leg.anchors[1], idx, used, prev_xy)
            ids = (gl.record.instance_id, r1.instance_id) if r1 is not None else None
            classes = (
                (gl.record.label, r1.label) if r1 is not None else None
            )
            goal = gl.geom  # (p0, p1) gate endpoints
            goal_mid = (
                ((goal[0][0] + goal[1][0]) / 2.0, (goal[0][1] + goal[1][1]) / 2.0)
                if goal is not None else None
            )
            rows.append({"kind": "corridor_between", "ids": ids, "classes": classes, "goal": goal_mid})
        else:
            kind = "via_near" if leg.kind is LegKind.VIA_NEAR else "goto"
            rows.append({
                "kind": kind,
                "ids": (gl.record.instance_id,),
                "classes": (gl.record.label,),
                "goal": gl.geom,
            })
    return rows


def _dist(a, b) -> float | None:
    if a is None or b is None:
        return None
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def run_parity(unity_root, questions_path, questions_dir, scenes=None) -> list[dict]:
    with open(questions_path, encoding="utf-8") as fh:
        data = json.load(fh)
    root = Path(unity_root)
    out: list[dict] = []
    for entry in data:
        scene_name = entry["scene"]
        if scenes is not None and scene_name not in scenes:
            continue
        folder = GB._find_scene_folder(root, scene_name)
        if folder is None:
            continue
        gt = load_scene(folder, scene_name=scene_name)
        idx = BasicSceneIndex(gt.instances)
        if_texts = entry["questions"].get("instruction_following", [])
        if not if_texts:
            continue
        traj, cands, frame, residual, pairs = GB._fit_scene_if_frame(
            gt, idx, if_texts, questions_dir
        )
        spawn_xy = None
        if frame is not None and pairs:
            start_pt = pairs[0][0][0, :2]
            mapped = frame.apply(np.asarray([start_pt], dtype=float))[0]
            spawn_xy = (float(mapped[0]), float(mapped[1]))
        frame_for_walls = (
            frame if residual is None or residual <= GB.WALL_FIT_MAX_RESIDUAL_M else None
        )
        wall_cells = GB._scene_wall_cells(gt, frame_for_walls)

        for qi, text in enumerate(if_texts):
            rub_rows = _rubric_side(text, gt, idx, spawn_xy)
            head_rows = _head_side(text, gt, idx, spawn_xy, wall_cells)
            n = max(len(rub_rows), len(head_rows))
            for li in range(n):
                rr = rub_rows[li] if li < len(rub_rows) else {"kind": None, "ids": None, "goal": None}
                hr = head_rows[li] if li < len(head_rows) else {"kind": None, "ids": None, "goal": None}
                d = _dist(rr.get("goal"), hr.get("goal"))
                same_ids = rr.get("ids") == hr.get("ids")
                row = {
                    "scene": scene_name,
                    "q_index": qi,
                    "leg": li,
                    "kind": rr.get("kind") or hr.get("kind"),
                    "rubric_ids": rr.get("ids"),
                    "rubric_classes": rr.get("classes"),
                    "rubric_goal": rr.get("goal"),
                    "head_ids": hr.get("ids"),
                    "head_classes": hr.get("classes"),
                    "head_goal": hr.get("goal"),
                    "goal_dist_m": round(d, 4) if d is not None else None,
                    "divergent": (not same_ids) or (rr.get("ids") is None) != (hr.get("ids") is None),
                }
                out.append(row)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="tools.resolver_parity",
        description="Audit resolve-outcome parity between the IF rubric and the "
        "navigation head's anchor resolution (issue #71).",
    )
    ap.add_argument("--groundtruth", required=True)
    ap.add_argument("--questions", default=str(GB.DEFAULT_QUESTIONS))
    ap.add_argument("--questions-dir", default=str(GB.DEFAULT_QUESTIONS_ROOT))
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--flag-dist-m", type=float, default=_FLAG_DIST_M_DEFAULT)
    args = ap.parse_args(argv)

    scenes = [s.strip() for s in args.scenes.split(",")] if args.scenes else None
    rows = run_parity(args.groundtruth, args.questions, args.questions_dir, scenes=scenes)

    n_legs = len(rows)
    n_divergent = sum(1 for r in rows if r["divergent"])
    n_dist_flagged = sum(
        1 for r in rows
        if r["goal_dist_m"] is not None and r["goal_dist_m"] >= args.flag_dist_m
    )
    print(f"resolver_parity: {n_legs} legs across {len({r['scene'] for r in rows})} scenes")
    print(f"  id-divergent legs: {n_divergent}")
    print(f"  goal_dist >= {args.flag_dist_m} m: {n_dist_flagged}")
    for r in rows:
        if r["divergent"] or (r["goal_dist_m"] is not None and r["goal_dist_m"] >= args.flag_dist_m):
            print(
                f"    {r['scene']} q{r['q_index']} leg{r['leg']} [{r['kind']}] "
                f"rubric={r['rubric_ids']} head={r['head_ids']} "
                f"dist={r['goal_dist_m']}"
            )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
