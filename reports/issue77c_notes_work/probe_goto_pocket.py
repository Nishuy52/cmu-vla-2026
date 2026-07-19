"""Issue #77c Phase 1b: generalized version of #79's probe_goto_pinch.py --
for a given (scene, qidx, leg_index) goto leg immediately following a
corridor_between leg, check whether the leg's own anchor cell EVER enters
the reachable set under the full pinch-relax schedule (through the leg's
OWN preceding gate). If it never does at any relax round -> disconnected by
real (uninflated) geometry, same finding as #79's hotel_room_2 case
(structural). If it does become reachable (or the relaxed nearest-reachable
point lands materially closer to the rubric goal than the plain BFS) ->
still-broken-fixable (a goal-resolution mechanism fix, #79's deferred #78
item, would help)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from core.groundtruth.loader import load_scene
import core.runner.gt_battery as GB
from core.perception.scene_index import BasicSceneIndex
from core.nav import planner as P
from core.geometry import toolbox as TB

ROOT = Path("../data/vla3d/Unity")


def probe(scene, qidx, leg_idx):
    with open(GB.DEFAULT_QUESTIONS, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = next(e for e in data if e["scene"] == scene)
    folder = GB._find_scene_folder(ROOT, scene)
    gt = load_scene(folder, scene_name=scene)
    idx = BasicSceneIndex(gt.instances)

    if_texts = entry["questions"]["instruction_following"]
    text = if_texts[qidx]

    if_traj, if_cands, frame, residual, pairs = GB._fit_scene_if_frame(
        gt, idx, if_texts, GB.DEFAULT_QUESTIONS_ROOT
    )
    spawn_xy = None
    if frame is not None and pairs and qidx < len(pairs):
        start_pt = pairs[qidx][0][0, :2]
        mapped = frame.apply(np.asarray([start_pt], dtype=float))[0]
        spawn_xy = (float(mapped[0]), float(mapped[1]))

    frame_for_walls = frame if residual is None or residual <= GB.WALL_FIT_MAX_RESIDUAL_M else None
    wall_cells = GB._scene_wall_cells(
        gt, frame_for_walls, unity_scenes_ros2_root=GB.DEFAULT_UNITY_SCENES_ROS2_ROOT
    )

    head, io, plan = GB._run_instruction_head(
        text, gt, idx, start_xy=spawn_xy, max_build_ticks=GB._IF_MAX_BUILD_TICKS,
        wall_cells=wall_cells,
    )
    if head is None or head._legs is None or leg_idx >= len(head._legs):
        print(f"{scene} q{qidx} leg{leg_idx}: could not build head/legs")
        return

    prev_leg = head._legs[leg_idx - 1]
    leg = head._legs[leg_idx]
    gate = prev_leg.geom
    rec = leg.record
    c = TB.P._as3(rec.centroid)
    anchor_xy = (float(c[0]), float(c[1]))

    leg_goals, *_rest = GB._if_rubric_geometry(text, gt, idx, start_xy=spawn_xy)
    rubric_xy = leg_goals[leg_idx][1] if leg_idx < len(leg_goals) else None

    cm = head._costmap
    pose = head._pose
    seen = cm.reachable_mask(pose)
    r, col = head.grid.world_to_cell(*anchor_xy)
    plain_hit = bool(seen[r, col]) if seen is not None else None
    plain_nrp = cm.nearest_reachable_point(anchor_xy, pose)
    plain_dist = float(np.hypot(plain_nrp[0] - rubric_xy[0], plain_nrp[1] - rubric_xy[1])) if rubric_xy else None

    print(f"=== {scene} q{qidx} leg{leg_idx} ===")
    print(f"  head_goal={leg.geom} rubric_goal={rubric_xy} gate={gate}")
    print(f"  plain: anchor_reachable={plain_hit} pocket={int(seen.sum())} nearest={plain_nrp} dist_to_rubric={plain_dist:.4f}" if plain_dist is not None else f"  plain: anchor_reachable={plain_hit}")

    best_dist = plain_dist
    ever_reachable = plain_hit
    relax_disc_m = P.PINCH_DISC_M
    for round_i in range(P.MAX_PINCH_RELAX_ROUNDS):
        pinch = P._pinch_costmap(
            cm, gate, pinch_disc_m=relax_disc_m,
            pinch_corridor_half_w_m=P.PINCH_CORRIDOR_HALF_W_M, start_xy=pose,
        )
        pseen = pinch.reachable_mask(pose)
        hit = bool(pseen[r, col]) if pseen is not None else None
        nrp = pinch.nearest_reachable_point(anchor_xy, pose)
        dist = float(np.hypot(nrp[0] - rubric_xy[0], nrp[1] - rubric_xy[1])) if rubric_xy else None
        print(f"  round {round_i} disc={relax_disc_m:.3f} pocket={int(pseen.sum())} anchor_reachable={hit} dist_to_rubric={dist:.4f}")
        if hit:
            ever_reachable = True
        if dist is not None and (best_dist is None or dist < best_dist):
            best_dist = dist
        relax_disc_m *= P.PINCH_RELAX_GROWTH

    verdict = "STRUCTURAL (never reachable, pinch relax doesn't help)" if not ever_reachable and (best_dist is None or best_dist >= plain_dist - 1e-6) else "POTENTIALLY FIXABLE"
    print(f"  VERDICT: {verdict} (plain_dist={plain_dist:.4f} best_dist={best_dist:.4f} ever_reachable={ever_reachable})")
    print()


if __name__ == "__main__":
    targets = [
        ("arabic_room", 1, 2),
        ("hotel_room_1", 1, 2),
        ("home_building_2", 1, 1),
        ("studio", 1, 2),
        ("hotel_room_2", 0, 1),
        ("livingroom_1", 1, 2),
    ]
    for scene, qidx, leg_idx in targets:
        try:
            probe(scene, qidx, leg_idx)
        except Exception as e:
            print(f"{scene} q{qidx} leg{leg_idx}: ERROR {e}")
            print()
