"""Stage 0 parts (2) and (3): pocket-excluded legs + currently-passing regression-risk
flags, built on top of mirror_audit.py's per-scene mirror + blocked-cell audit."""
import json
import sys
sys.path.insert(0, "/tmp/claude-1000/-home-jason-cmu-ws/042376ea-8ed2-4b27-b155-0f949ec316c4/scratchpad")
import mirror_audit as M
import core.runner.gt_battery as GB
from core.groundtruth.loader import load_scene
from core.perception.scene_index import BasicSceneIndex

RESULTS = M.RESULTS
ROOT = M.ROOT

audit = json.load(open("/tmp/claude-1000/-home-jason-cmu-ws/042376ea-8ed2-4b27-b155-0f949ec316c4/scratchpad/all_scenes.json"))


def leg_rows_for_scene(scene):
    return [r for r in RESULTS["scores"] if r["scene"] == scene and r["qtype"] == "instruction_following"]


def pocket_excluded_legs(scene, mir):
    """For each leg's own anchor goal (leg_probe.our_goal), is it outside the spawn
    BFS pocket (reachable_mask over the INFLATED mirror costmap from the shared
    scene spawn)? Matches build_miss_table's arrival-blocked-pool lens: pocket
    exclusion is a structural reason a leg can never be reached offline, independent
    of tolerance."""
    cm = mir["cm"]
    spawn_xy = mir["spawn_xy"]
    reach = cm.reachable_mask(spawn_xy)
    out = []
    for r in leg_rows_for_scene(scene):
        if not r.get("leg_probe"):
            continue
        for leg in r["leg_probe"]:
            gx, gy = leg["our_goal"]
            rr, cc = mir["grid"].world_to_cell(gx, gy)
            in_pocket = (
                reach is not None
                and 0 <= rr < reach.shape[0] and 0 <= cc < reach.shape[1]
                and reach[rr, cc]
            )
            out.append(dict(
                question=r["question"][:60], leg=leg["i"], kind=leg["kind"],
                goal=[gx, gy], pocket_excluded=not bool(in_pocket),
            ))
    return out


def regression_risk_legs(scene, mir, scene_audit, proximity_m=0.6):
    """For each CURRENTLY-PASSING leg (reached_in_order True), replan the leg's own
    question (GB._drive_if_path) and flag it if the planned route passes within
    ``proximity_m`` of a cell this scene's audit found to be OBB-clearable (i.e. an
    object-box blocker that OBB rasterization is about to remove) -- a route that
    threads near/around such a cell today may replan differently once that cell
    stops being solid (regression risk, risk-register row 'Carve reroutes a passing
    leg', now generalized to the OBB-rasterization lane)."""
    gt = mir["gt"]
    idx = mir["idx"]
    wall_cells = mir["wall_cells"]
    grid = mir["grid"]

    # Recover the actual OBB-clearable object cells (not just the count) by
    # re-deriving them the same way audit_scene did.
    clearable_cells = []
    cm = mir["cm"]
    frame = mir["frame"]
    seen = set()
    for traj in mir["if_traj"]:
        if traj is None or traj.shape[0] == 0 or frame is None:
            continue
        mapped = frame.apply(traj[:, :2])
        for x, y in mapped:
            r, c = grid.world_to_cell(float(x), float(y))
            if not (0 <= r < cm.base_blocked.shape[0] and 0 <= c < cm.base_blocked.shape[1]):
                continue
            if not cm.base_blocked[r, c]:
                continue
            if (r, c) in seen:
                continue
            seen.add((r, c))
            cx, cy = grid.cell_to_world(r, c)
            kind, labels = M.classify_cell_blocker(cx, cy, mir)
            if kind == "object" and labels and all(
                (cl is True) for _, _, cl, _ in labels if cl is not None
            ):
                clearable_cells.append((cx, cy))

    out = []
    if not clearable_cells:
        return out

    rows = leg_rows_for_scene(scene)
    for r in rows:
        if not r.get("leg_outcomes"):
            continue
        passing = all(leg["reached_in_order"] for leg in r["leg_outcomes"])
        thread_ok = (r.get("n_threading_violations") or 0) == 0
        avoid_ok = (r.get("n_avoid_violations") or 0) == 0
        if not (passing and thread_ok and avoid_ok):
            continue
        path = GB._drive_if_path(
            r["question"], gt, idx, start_xy=mir["spawn_xy"], wall_cells=wall_cells
        )
        if path.shape[0] == 0:
            continue
        near = []
        for cx, cy in clearable_cells:
            d = ((path[:, 0] - cx) ** 2 + (path[:, 1] - cy) ** 2) ** 0.5
            if d.min() <= proximity_m:
                near.append((float(d.min()), (cx, cy)))
        if near:
            near.sort()
            out.append(dict(
                question=r["question"][:70], rubric=r["rubric_score"],
                n_near_clearable_cells=len(near),
                closest_m=round(near[0][0], 3),
            ))
    return out


def main():
    result = {}
    for scene in M.SCENES:
        mir = M.build_scene_mirror(scene)
        pocket = pocket_excluded_legs(scene, mir)
        n_excluded = sum(1 for p in pocket if p["pocket_excluded"])
        risk = regression_risk_legs(scene, mir, audit.get(scene, {}))
        result[scene] = dict(
            n_legs_probed=len(pocket),
            n_pocket_excluded=n_excluded,
            pocket_legs=pocket,
            regression_risk=risk,
        )
        print(scene, "legs", len(pocket), "pocket_excluded", n_excluded,
              "regression_risk", len(risk), file=sys.stderr)
    json.dump(result, open(
        "/tmp/claude-1000/-home-jason-cmu-ws/042376ea-8ed2-4b27-b155-0f949ec316c4/scratchpad/pocket_regression.json",
        "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
