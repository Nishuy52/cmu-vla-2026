"""Battery v2 — REAL accuracy scoring against ground-truth VLA-3D scenes.

Where :mod:`core.runner.battery` scores structural health over *synthetic* scenes (no
ground truth exists for the training questions), this harness loads the actual VLA-3D
scene folders and scores each question for accuracy:

* NUMERICAL           -> exact-match count + independent second opinion
* OBJECT_REFERENCE    -> 3D IoU of our box vs the GT target box
* INSTRUCTION_FOLLOWING -> discrete Frechet + fraction of GT path within 1.0 m

The scene is *fully observed*: every GT object is an :class:`InstanceRecord` with
``n_obs=3``, so the answer heads resolve directly with no exploration needed. For
NUMERICAL/OBJECT_REFERENCE the scorers call our resolver on the GT index directly
(fast, deterministic). For INSTRUCTION_FOLLOWING we drive the full pipeline via
:func:`core.runner.single.run_question` over a GT-mirrored :class:`MockRobotIO` so the
instruction head plans a real waypoint path through a costmap built from the GT
geometry; the emitted waypoints are our path, scored against ``trajectory_qN.ply``.

Scene discovery: given ``--groundtruth <unity_root>``, we match each ``questions.json``
scene name to a subfolder of the same name (VLA-3D uses a flat ``<scene>/`` layout that
matches the challenge scene names verbatim — see docs/vla3d_notes.md §4). Scenes with no
folder present are skipped (reported as ``missing``), so a partial download still yields
a partial report.

Output: ``reports/gt_battery_<date>/`` with ``gt_battery_report.md`` (per-type accuracy
+ per-question table with failure notes) and ``gt_battery_results.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from itertools import product
from pathlib import Path

import numpy as np

from core.groundtruth import scoring as S
from core.groundtruth.loader import GTScene, load_scene
from core.interfaces import QType, WaypointCmd
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.mocks.synthetic_scene import (
    FLOOR_SPACING,
    TERRAIN_SLAB_MAX_Z,
    Room,
    SyntheticScene,
)
from core.perception.scene_index import BasicSceneIndex
from core.runner.provenance import collect_provenance

_SRC = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = (
    _SRC.parent / "upstream" / "CMU-VLN-Challenge-2026" / "questions" / "questions.json"
)
DEFAULT_QUESTIONS_ROOT = DEFAULT_QUESTIONS.parent
DEFAULT_OUT_ROOT = _SRC.parent / "reports"
#: True numerical answer key (arch-F3): the human answers extracted from each scene's
#: questions.pdf text layer — the PRIMARY numerical yardstick, replacing the
#: pipeline-self-consistency proxy. Absent file -> battery still runs, true fields null.
DEFAULT_ANSWERS = _SRC.parent / "docs" / "gt_answers_numerical.json"
#: Root holding, per scene, ``<scene>/<scene>/traversable_area.ply`` — the ROS2/sim-side
#: traversable-floor mesh used to derive interior walls for the IF mirror costmap
#: (IF-F2 wall realism; see ``_scene_wall_cells``). Not part of the VLA-3D Unity root
#: (which ships only per-region AABBs, no mesh) — a sibling dataset in the same sim/
#: trajectory frame as ``trajectory_qN.ply``, so it needs the same sim->object
#: :class:`~core.groundtruth.scoring.Frame2D` fit before it can be rasterized into the
#: object-frame costmap.
DEFAULT_UNITY_SCENES_ROS2_ROOT = _SRC.parent / "data" / "unity_scenes_ros2"

# The instruction-following trajectory files sit at questions/<scene>/trajectory_q{4,5}.ply.
# questions.json order is 1 numerical, 2 object_reference, 2 instruction_following, so the
# two IF questions map to q4 and q5 respectively (docs/vla3d_notes.md §5).
_IF_TRAJ_INDEX = {0: 4, 1: 5}

_BATTERY_TICK_HZ = 1.0


# --------------------------------------------------------------------------- scene mirror


def _gt_footprint_bounds(gt: GTScene, pad: float) -> tuple[float, float, float, float]:
    """The mirror costmap's outer rectangle: GT instance AABBs' footprint + ``pad``."""
    mins = np.array([r.aabb_min for r in gt.instances])
    maxs = np.array([r.aabb_max for r in gt.instances])
    x0 = float(mins[:, 0].min()) - pad
    y0 = float(mins[:, 1].min()) - pad
    x1 = float(maxs[:, 0].max()) + pad
    y1 = float(maxs[:, 1].max()) + pad
    return x0, y0, x1, y1


#: Issue #53 — a GT instance whose raw AABB footprint spans more than this fraction of
#: the SCENE'S OWN room footprint in BOTH x and y is "room-scale": VLA-3D sometimes
#: records an architectural aggregate (a room's whole wall shell, or an unclassified
#: scene-level mesh labelled "unknown") as one bounding box rather than the real thin
#: element it stands in for. Stamping that box solid seals most/all of the room's
#: interior — the real wall is thin, the AABB just happens to span the room. Purely
#: geometric and self-referential per scene (a fraction of THAT scene's own footprint,
#: never a fixed metres constant), so it needs no cross-scene tuning: swept over all 15
#: GT scenes, every floor-level instance clearing this bar in BOTH axes is labelled
#: "wall", "floor", or "unknown" (home_building_1 id 25/82, home_building_2 id 126/132/
#: 167, hotel_room_2 id 89, japanese_room id 12, livingroom_2 id 10, studio id 31) —
#: zero real furniture instance across those scenes clears it (the closest, a hotel_
#: room_2 bed frame, sits at 0.38/0.28 — short on the y axis). Objects like this are
#: skipped by :func:`_synthetic_from_gt` rather than stamped: the traversable-mesh-
#: derived ``wall_cells`` (IF-F2, already wired) become the sole source of interior-wall
#: geometry for the room, and the room's own outer boundary (``SyntheticScene._is_wall``)
#: still seals the exterior — avoiding a double representation where the same walls are
#: both a raw solid AABB AND mesh-derived wall cells.
ARCHITECTURAL_AABB_ROOM_FRACTION: float = 0.3


def _is_architectural_room_scale_aabb(
    rec_aabb_min: np.ndarray, rec_aabb_max: np.ndarray, room_w: float, room_h: float
) -> bool:
    """True if ``rec``'s footprint is room-scale in both axes AND floor-level.

    See :data:`ARCHITECTURAL_AABB_ROOM_FRACTION`. Floor-level (not an overhang) uses
    the same :data:`~core.mocks.synthetic_scene.TERRAIN_SLAB_MAX_Z` cutoff the terrain
    mirror itself uses to tell floor obstacles from tabletop/ceiling overhangs — an
    elevated room-scale slab (a real ceiling) already reads as an overhang and was
    never the problem.
    """
    cz = float(rec_aabb_min[2])
    if cz >= TERRAIN_SLAB_MAX_Z:
        return False
    sx = float(rec_aabb_max[0] - rec_aabb_min[0])
    sy = float(rec_aabb_max[1] - rec_aabb_min[1])
    if room_w <= 0 or room_h <= 0:
        return False
    return (
        sx / room_w > ARCHITECTURAL_AABB_ROOM_FRACTION
        and sy / room_h > ARCHITECTURAL_AABB_ROOM_FRACTION
    )


def _synthetic_from_gt(
    gt: GTScene, pad: float = 1.5, *, wall_cells: set[tuple[int, int]] | None = None
) -> SyntheticScene:
    """Build a SyntheticScene that mirrors the GT AABBs (for the IF costmap).

    One room bounding all GT footprints (plus padding); one box per GT instance at its
    AABB footprint centre + size. This gives the instruction head a terrain/costmap
    consistent with the geometry the heads resolve against. ``wall_cells`` (IF-F2 wall
    realism) additionally stamps interior walls derived off the scene's
    ``traversable_area.ply`` (see :func:`_scene_wall_cells`) — ``None`` reproduces the
    old boundary-only behaviour (object obstacles + outer boundary, no interior walls).

    Room-scale architectural AABBs (issue #53 — see
    :data:`ARCHITECTURAL_AABB_ROOM_FRACTION`) are NOT stamped as solid boxes: the raw
    box is a VLA-3D bounding-box-of-an-aggregate, not the real thin element, and
    stamping it solid seals interior area the room-scale box only happens to cover.
    """
    x0, y0, x1, y1 = _gt_footprint_bounds(gt, 0.0)
    room_w, room_h = x1 - x0, y1 - y0

    sc = SyntheticScene(0, extra_wall_cells=wall_cells)
    # Shift into non-negative coords is unnecessary — Room accepts arbitrary bounds.
    sc.rooms = [Room(x0 - pad, y0 - pad, x1 + pad, y1 + pad)]
    sc._split_x = None
    sc.doorway = None
    sc.objects = []
    for rec in gt.instances:
        if _is_architectural_room_scale_aabb(rec.aabb_min, rec.aabb_max, room_w, room_h):
            continue
        cx = float((rec.aabb_min[0] + rec.aabb_max[0]) / 2)
        cy = float((rec.aabb_min[1] + rec.aabb_max[1]) / 2)
        sx = float(max(rec.aabb_max[0] - rec.aabb_min[0], 0.05))
        sy = float(max(rec.aabb_max[1] - rec.aabb_min[1], 0.05))
        sz = float(max(rec.aabb_max[2] - rec.aabb_min[2], 0.05))
        # Carry the true base height so the terrain mirror can distinguish floor
        # obstacles from tabletop/wall-mounted overhangs (a plant ON a cabinet has
        # base ~0.83 m; the floor beside it is drivable). Without this every AABB was
        # stamped floor-to-top, sealing floor near most leg anchors (T11 sig-1/2).
        cz = float(rec.aabb_min[2])
        sc.place_box(rec.label, cx, cy, sx, sy, sz, cz=cz)
    return sc


#: Wall-cell grid resolution (m) — matches the terrain/occupancy lattice
#: (``FLOOR_SPACING`` / ``core.nav.occupancy.CELL_M``) so derived wall cells line up
#: exactly with the costmap cells the planner reasons over.
_WALL_CELL_M: float = FLOOR_SPACING
#: Dilation radius (cells) applied to traversable-point coverage before the complement
#: is called "wall" — bridges point-cloud sampling gaps (the mesh is a scattered sample,
#: not a filled raster) without erasing real doorways. At ``_WALL_CELL_M=0.1`` this is a
#: 0.2 m pad from each covered cell; the shipped meshes sample at millimetre to
#: low-centimetre spacing (dense), so this comfortably bridges sampling gaps while
#: staying well under the width of a real doorway. Deliberately kept small: it is not
#: sized to also absorb sim->object frame-fit residual (up to the 1.0 m alignment gate,
#: see ``S._ALIGN_RESIDUAL_GATE_M``) — a scene whose fit is noisy can still wall off a
#: real (but narrow) doorway near the fit's error radius. See the wall-realism note in
#: the report for known caveats; ``--no-walls`` is the escape when a scene's costmap
#: looks wrong.
_WALL_DILATE_CELLS: int = 2

#: Wall-derivation frame-fit gate (issue #52) — TIGHTER than the scoring alignment gate
#: (``S._ALIGN_RESIDUAL_GATE_M`` = 1.0 m). A scene whose sim->object fit residual clears
#: 1.0 m but sits in the 0.8-1.0 m band is aligned enough to trust for Frechet/coverage
#: scoring (a smooth diagnostic that degrades gracefully with error) but NOT enough to
#: rasterize the traversable mesh into hard wall cells (a binary decision that can wall
#: off a real object location on a marginal fit — observed at livingroom_1's 0.926 m
#: residual, which misprojected walls onto a real position and flipped an IF question
#: 0.5->0.0 for measurement, not planning, reasons). Declining to derive walls (falling
#: back to the boundary-only costmap) beats fabricating misplaced ones — spec-over-
#: sample per the generalization protocol (docs/calibration.md); registered there too.
WALL_FIT_MAX_RESIDUAL_M: float = 0.8


def _traversable_ply_path(scene_name: str, root: os.PathLike | str | None = None) -> Path:
    base = Path(root) if root is not None else DEFAULT_UNITY_SCENES_ROS2_ROOT
    return base / scene_name / scene_name / "traversable_area.ply"


def _derive_wall_cells(
    traversable_xy: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    cell_m: float = _WALL_CELL_M,
    dilate_cells: int = _WALL_DILATE_CELLS,
) -> set[tuple[int, int]]:
    """Rasterize traversable-mesh XY points to the wall-cell lattice and return the
    cells INSIDE ``[x0, x1] x [y0, y1]`` that are NOT covered (after dilation).

    Pure geometry, no I/O — ``traversable_xy`` is already in the destination (object)
    frame. Grid cells use the absolute ``round(coord / cell_m)`` lattice (see
    ``SyntheticScene._in_extra_wall_cells``), so the returned set lines up with the
    terrain grid without needing a shared origin.
    """
    ix0 = int(round(x0 / cell_m))
    ix1 = int(round(x1 / cell_m))
    iy0 = int(round(y0 / cell_m))
    iy1 = int(round(y1 / cell_m))
    nx = ix1 - ix0 + 1
    ny = iy1 - iy0 + 1
    if nx <= 0 or ny <= 0:
        return set()

    free = np.zeros((nx, ny), dtype=bool)
    pts = np.asarray(traversable_xy, dtype=float)
    if pts.size:
        pix = np.round(pts[:, 0] / cell_m).astype(np.int64)
        piy = np.round(pts[:, 1] / cell_m).astype(np.int64)
        sel = (pix >= ix0) & (pix <= ix1) & (piy >= iy0) & (piy <= iy1)
        free[pix[sel] - ix0, piy[sel] - iy0] = True

    if dilate_cells > 0:
        dilated = free.copy()
        for dx in range(-dilate_cells, dilate_cells + 1):
            for dy in range(-dilate_cells, dilate_cells + 1):
                if dx == 0 and dy == 0:
                    continue
                sx0, sx1 = max(0, -dx), nx - max(0, dx)
                dx0, dx1 = max(0, dx), nx - max(0, -dx)
                sy0, sy1 = max(0, -dy), ny - max(0, dy)
                dy0, dy1 = max(0, dy), ny - max(0, -dy)
                if sx1 <= sx0 or sy1 <= sy0:
                    continue
                dilated[dx0:dx1, dy0:dy1] |= free[sx0:sx1, sy0:sy1]
        free = dilated

    rows, cols = np.nonzero(~free)
    return {(int(ix0 + r), int(iy0 + c)) for r, c in zip(rows.tolist(), cols.tolist())}


def _scene_wall_cells(
    gt: GTScene,
    frame: "S.Frame2D | None",
    *,
    pad: float = 1.5,
    unity_scenes_ros2_root: os.PathLike | str | None = None,
) -> set[tuple[int, int]] | None:
    """Derive interior-wall grid cells for ``gt``'s mirror costmap (IF-F2 wall realism).

    Reads the scene's ``traversable_area.ply`` (sim/trajectory frame), maps it into the
    object frame via the scene's fitted sim->object ``frame`` (the same
    :class:`~core.groundtruth.scoring.Frame2D` the IF trajectory scoring uses), and
    rasterizes the coverage complement inside the mirror costmap's outer rectangle
    (:func:`_gt_footprint_bounds`) to wall cells (:func:`_derive_wall_cells`).

    Returns ``None`` — the caller then reproduces the old boundary-only costmap — when
    there is no fitted frame to align the mesh with (no IF question resolved a terminal
    goal, or the scene is a confirmed frame-fit failure) or the scene ships no
    ``traversable_area.ply``.
    """
    if frame is None:
        return None
    ply_path = _traversable_ply_path(gt.scene_name, unity_scenes_ros2_root)
    if not ply_path.exists():
        return None
    pts = S.load_trajectory_ply(ply_path)
    if pts.size == 0:
        return None
    mapped = frame.apply(pts[:, :2])
    x0, y0, x1, y1 = _gt_footprint_bounds(gt, pad)
    return _derive_wall_cells(mapped, x0, y0, x1, y1)


_IF_MAX_BUILD_TICKS = 12  # ticks to let the instruction head ground legs + plan the route


def _drive_if_path(
    text: str,
    gt: GTScene,
    idx: BasicSceneIndex,
    *,
    max_build_ticks: int = _IF_MAX_BUILD_TICKS,
    start_xy: tuple[float, float] | None = None,
    wall_cells: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Plan an instruction-following path over the GT scene and return it as (N, 2).

    The scene is *fully observed*, so we do NOT run the FSM's explore/watchdog loop
    (which would spin on terrain recomputation with no new information). Instead we
    drive the :class:`InstructionHead` directly for a few ticks — just enough to
    ingest terrain, ground the route legs against the GT index, and plan the path
    through the costmap — then return the head's fully planned route (the
    ``BreadcrumbFollower.path``). This is our system's intended trajectory for the
    instruction, which is what the GT ``trajectory_qN.ply`` is compared against.

    ``start_xy`` sets the robot spawn in the GT (object) frame. When the scene frame
    has been fitted, the battery passes the GT trajectory's shared start mapped into
    the object frame, so our planned path departs from the same spawn the GT path does
    (a fair like-for-like comparison). Absent a fit it defaults to the scene corner.
    """
    from core.parsing.regex_tier import parse_regex
    from core.heads.instruction import InstructionHead

    plan = parse_regex(text)
    if plan.qtype is not QType.INSTRUCTION_FOLLOWING or not plan.route:
        return np.empty((0, 2), dtype=float)

    sc = _synthetic_from_gt(gt, wall_cells=wall_cells)
    clk = FakeClock(0.0)
    if start_xy is not None:
        start_x, start_y = float(start_xy[0]), float(start_xy[1])
    else:
        start_x = float(min(r.aabb_min[0] for r in gt.instances)) + 0.5
        start_y = float(min(r.aabb_min[1] for r in gt.instances)) + 0.5
    io = MockRobotIO(sc, clk, start_x=start_x, start_y=start_y)

    head = InstructionHead(plan=plan)
    for _ in range(max_build_ticks):
        head.advance(io, idx)
        clk.advance(1.0)
        if head._follower is not None and head._follower.path:
            break

    follower = head._follower
    if follower is None or not follower.path:
        # fall back to any streamed waypoints if the follower never firmed up
        if io.waypoints:
            return np.array([[w.x, w.y] for w in io.waypoints], dtype=float)
        return np.empty((0, 2), dtype=float)
    return np.array([[p[0], p[1]] for p in follower.path], dtype=float)


#: Constant-speed kinematic-follower step (m) per breadcrumb tick when simulating the
#: DRIVEN trajectory (IF-F2). v1 simplification: the vehicle advances straight toward
#: each published crumb at a fixed step, no dynamics/heading lag. Documented as a v1
#: proxy — good enough to test ordered-leg arrival, gate threading, and avoid breaches,
#: which is what the rubric scores; it does NOT model local-planner deviation or
#: waypoint-snapping (those are Ubuntu-sim concerns).
_DRIVE_STEP_M: float = 0.25
_DRIVE_MAX_TICKS: int = 4000  # hard cap so a stuck follower can't loop forever
#: Net-progress stall guard for the driven-sim: if the vehicle moves less than
#: ``_DRIVE_STALL_EPS_M`` over ``_DRIVE_STALL_TICKS`` consecutive steps it is wedged and
#: the drive ends (score what was driven). Sized so a genuinely slow-but-moving vehicle
#: (>= one step every ~40 ticks) is never cut, but a true wedge stops promptly.
_DRIVE_STALL_EPS_M: float = 0.05
_DRIVE_STALL_TICKS: int = 40

#: Max number of full ``head.advance`` re-ticks DURING the drive (each re-integrates
#: terrain + re-grounds + can re-plan, ~0.5 s, so it must be bounded). The head is re-ticked
#: only while the committed route can still grow (a later leg still ungrounded / a provisional
#: terminal withheld); once this budget is spent — or the route covers every leg — we drive
#: the follower directly (microseconds/step). In the fully-observed battery the route commits
#: on tick 0, so this budget is rarely touched; it exists so a route that can NEVER fully
#: ground (a leg unresolvable from the mirror scene) cannot make the drive loop 4000× at the
#: head's per-tick cost. Matches the build budget: the drive gets the same extension window.
_DRIVE_HEAD_RETICK_BUDGET: int = _IF_MAX_BUILD_TICKS


def _run_instruction_head(
    text: str,
    gt: GTScene,
    idx: BasicSceneIndex,
    *,
    start_xy: tuple[float, float] | None,
    max_build_ticks: int,
    wall_cells: set[tuple[int, int]] | None = None,
):
    """Build a scene mirror + MockRobotIO and tick the InstructionHead until its route
    firms up. Returns ``(head, io, plan)`` (head is None when the question isn't IF)."""
    from core.parsing.regex_tier import parse_regex
    from core.heads.instruction import InstructionHead

    plan = parse_regex(text)
    if plan.qtype is not QType.INSTRUCTION_FOLLOWING or not plan.route:
        return None, None, plan

    sc = _synthetic_from_gt(gt, wall_cells=wall_cells)
    clk = FakeClock(0.0)
    if start_xy is not None:
        start_x, start_y = float(start_xy[0]), float(start_xy[1])
    else:
        start_x = float(min(r.aabb_min[0] for r in gt.instances)) + 0.5
        start_y = float(min(r.aabb_min[1] for r in gt.instances)) + 0.5
    io = MockRobotIO(sc, clk, start_x=start_x, start_y=start_y)

    head = InstructionHead(plan=plan)
    for _ in range(max_build_ticks):
        head.advance(io, idx)
        clk.advance(1.0)
        if head._follower is not None and head._follower.path:
            break
    return head, io, plan


def _drive_if_trajectory(
    text: str,
    gt: GTScene,
    idx: BasicSceneIndex,
    *,
    max_build_ticks: int = _IF_MAX_BUILD_TICKS,
    start_xy: tuple[float, float] | None = None,
    wall_cells: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Simulate the DRIVEN trajectory (IF-F2), returning the pose stream as (N, 2).

    Unlike :func:`_drive_if_path` (which returns the *planned* ``BreadcrumbFollower.path``),
    this closes the loop against the instruction head: we build + ground the route, then
    step a constant-speed kinematic vehicle toward the follower's current crumb, recording
    each pose — and we KEEP TICKING THE HEAD between drive steps so the committed route
    re-grounds and EXTENDS as later legs commit (H3c prefix-growth / H4c provisional-terminal
    commit). The follower's progress index therefore advances in lockstep with the *real*
    (moving) pose. This is what the rubric proxy scores — the actual trajectory the vehicle
    followed through the crumbs, so ordered arrival / gate threading / avoid breaches are
    measured on motion, not on a plan. v1 kinematics: straight steps of ``_DRIVE_STEP_M``
    toward the crumb (see the constant's note).

    HARNESS-bug history (the reason this was rewritten): the previous version ticked the
    head ``max_build_ticks`` times from a *stationary* spawn to firm up the route, then
    drove the SAME follower object. But every build tick already calls
    ``head.advance -> _drive -> follower.advance`` at the spawn pose, so for a short route
    sitting near the spawn the follower's progress index was fully consumed before the
    drive began. The reused follower returned ``None`` on the first drive step -> a 2-pose
    trajectory (spawn + terminal vertex) that reaches no leg -> rubric 0. That was the
    ``poses=2..30`` signature in the report: a harness artifact, not a pipeline failure.
    We fix it by (a) rewinding the follower's progress after the build, and (b) driving
    closed-loop with a moving pose so progress only advances as the vehicle really moves.

    Performance: re-ticking ``head.advance`` (terrain re-integration + full re-ground +
    re-plan) costs ~0.5 s/tick, so we do it ONLY while the committed route can still grow
    (``_committable_prefix_len`` has not reached every leg, i.e. a later leg is still
    ungrounded or a provisional terminal is withheld). In this fully-observed battery the
    route commits on tick 0, so we fall straight through to driving the follower directly
    (microseconds/step). When exploration/provisional-withholding is in play the head keeps
    ticking until the route is whole, which is exactly the general-case fidelity F2 wants.
    """
    head, io, plan = _run_instruction_head(
        text, gt, idx, start_xy=start_xy, max_build_ticks=max_build_ticks,
        wall_cells=wall_cells,
    )
    if head is None:
        return np.empty((0, 2), dtype=float)
    follower = head._follower
    if follower is None or not follower.path:
        return np.empty((0, 2), dtype=float)

    # The build loop drove the follower from the stationary spawn (each _run tick calls
    # head.advance -> _drive), so its progress index may already be advanced (or fully
    # consumed) against the spawn pose. Rewind so the drive starts at the route's head and
    # progresses only as the vehicle really moves.
    follower._idx = 0
    follower._hist = []
    follower._last_crumb_idx = -1

    odom = io.latest_odom()
    pose = (float(odom.x), float(odom.y)) if odom is not None else (0.0, 0.0)
    t = 0.0
    poses: list[tuple[float, float]] = [pose]
    last_term = (float(follower.path[-1][0]), float(follower.path[-1][1]))
    n_legs = len(head._legs) if head._legs else len(plan.route)
    head_reticks_left = _DRIVE_HEAD_RETICK_BUDGET
    # Stall guard: if the vehicle makes no net progress over a window of steps it is wedged
    # against an obstacle (the crumb sits across a corner the straight kinematic step can't
    # round) — driving on cannot help, so stop and score the trajectory so far instead of
    # padding it to the watchdog length. Without this a wedge produced watchdog-length
    # (poses=4002) rows that only inflated runtime, never arrival.
    stall_ref = pose
    stall_ticks = 0

    for _ in range(_DRIVE_MAX_TICKS):
        # Route still growing? Re-tick the head (moving the vehicle first) so re-grounding,
        # prefix-growth and provisional-terminal commit can extend the committed route. Once
        # the route covers every leg — or the bounded re-tick budget is spent (a leg that
        # never grounds must not make us pay the head's ~0.5 s/tick cost 4000×) — we stop
        # re-ticking and drive the committed follower directly.
        route_growing = head._driven_prefix < n_legs and head_reticks_left > 0
        if route_growing:
            head_reticks_left -= 1
            io.set_pose(pose[0], pose[1])
            head.advance(io, idx)
            new_follower = head._follower
            if new_follower is not None and new_follower.path:
                if new_follower is not follower:
                    # The route was extended/replanned: resync progress to the nearest
                    # not-yet-passed vertex so the drive continues smoothly on the new path.
                    follower = new_follower
                    follower._idx = _nearest_forward_idx(follower.path, pose)
                    follower._hist = []
                    follower._last_crumb_idx = -1
                last_term = (
                    float(follower.path[-1][0]),
                    float(follower.path[-1][1]),
                )

        wp = follower.advance(pose, t)
        if wp is None:
            break
        target = (float(wp.x), float(wp.y))
        dx, dy = target[0] - pose[0], target[1] - pose[1]
        d = (dx * dx + dy * dy) ** 0.5
        if d <= _DRIVE_STEP_M:
            pose = target
        else:
            pose = (pose[0] + _DRIVE_STEP_M * dx / d, pose[1] + _DRIVE_STEP_M * dy / d)
        poses.append(pose)
        t += 1.0

        # Net-progress stall guard (see stall_ref note above).
        if ((pose[0] - stall_ref[0]) ** 2 + (pose[1] - stall_ref[1]) ** 2) > (
            _DRIVE_STALL_EPS_M**2
        ):
            stall_ref = pose
            stall_ticks = 0
        else:
            stall_ticks += 1
            if stall_ticks >= _DRIVE_STALL_TICKS:
                break

    # Ensure the planned terminal vertex is represented (the follower returns None once the
    # progress index passes the last crumb, which can be a step short of the exact vertex
    # under the constant-speed stepping).
    if not poses or (poses[-1][0] - last_term[0]) ** 2 + (
        poses[-1][1] - last_term[1]
    ) ** 2 > (_DRIVE_STEP_M**2):
        poses.append((float(last_term[0]), float(last_term[1])))
    return np.array(poses, dtype=float)


def _nearest_forward_idx(
    path: list[tuple[float, float]], pose: tuple[float, float]
) -> int:
    """Index of the path vertex nearest to ``pose`` (used to resync progress after the
    committed route is extended/replanned mid-drive)."""
    best_i, best_d = 0, float("inf")
    for i, p in enumerate(path):
        d = (p[0] - pose[0]) ** 2 + (p[1] - pose[1]) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


#: issue #61 — clearance (m) applied when pushing a GOTO/VIA_NEAR leg goal off a
#: floor-level obstacle footprint it falls inside (see :func:`_nearest_free_goal`).
#: Matches ``core.nav.costmap.VEHICLE_RADIUS_M`` — the same "the vehicle's own body
#: can't be here" margin the real navigation stack already inflates every obstacle
#: by, so the rubric goal and the drivable target agree on what counts as clear.
_RUBRIC_GOAL_CLEARANCE_M: float = 0.4
#: Bounded push distance (m): a local, geometric correction only (get the goal off
#: the ONE object it's sitting on/inside), never an unbounded drift chasing some
#: notion of "the right approach side" — if bounded pushing can't clear every
#: overlapping footprint, the raw centroid stands (an honest near-miss beats a
#: silently fabricated goal position).
_RUBRIC_GOAL_MAX_PUSH_M: float = 2.5


def _nearest_free_goal(
    xy: tuple[float, float],
    gt: GTScene,
    *,
    anchor_id: int | None = None,
    approach_xy: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Push a raw anchor centroid off any floor-level obstacle footprint it falls
    inside (issue #61; issue #66 extends this to the anchor's OWN footprint).

    An anchor resolved from a clause like "the magazine ON the ottoman" has its own
    raw centroid sitting squarely inside the SUPPORTER's floor footprint (the
    magazine sits on top of the ottoman) — a point the vehicle physically cannot
    occupy, and not where a real path would stop (it stops at the supporter's
    edge). ``_if_rubric_geometry`` previously used the bare, unprojected centroid as
    the rubric's arrival target, unlike ``InstructionHead._goto_point``/``_via_point``
    (real navigation), which always project onto free/reachable space — so the
    rubric could require arrival at a point our own drive would never plan to (or
    could reach at all). This mirrors that projection with a lightweight, pose-free
    version: push off any floor-level, non-architectural-room-scale instance
    footprint the point falls inside, iterating a bounded number of times for a
    nested/overlapping case.

    Issue #66: earlier this excluded the resolved anchor's OWN instance from the
    push (only pushing off OTHER objects), on the theory that "go to X" should
    target X's centroid. But X is itself real, solid, stamped furniture whenever
    its footprint is large enough to register as an obstacle (``_synthetic_from_gt``
    stamps every non-architectural instance, including the leg's own
    target) — ``_goto_point``/``_via_point`` never target a point inside X's own
    solid geometry either, they BFS/gradient onto the nearest free cell around it.
    A goal left at X's raw centroid is arrival-tolerance-unreachable by construction
    for any X whose footprint half-diagonal exceeds ``ARRIVAL_TOL_M`` (traced at
    e.g. a bench, a guitar, a set of stairs, a soccer ball — see issue #66's
    classification table) — not a stamping-fidelity defect, a goal-definition one.
    So the anchor's own footprint is no longer skipped — but pushing it needs to
    pick the correct SIDE, not just the geometrically nearest edge: a plain
    "nearest of 4 edges" choice ignores which side is actually approached (real
    furniture usually has one accessible side; the near edge of a square footprint
    is a coin-flip that traced worse against both the driven path AND the GT
    reference path in practice). When ``anchor_id``/``approach_xy`` are given (the
    leg's own resolved instance and the point the route arrives FROM — the
    previous leg's goal, or the scene spawn for the first leg), the anchor's own
    push is instead directional: pushed along whichever axis the approach point is
    furthest offset on, toward the side the approach point is actually on —
    mirroring "walk up to X from where you are and stop at its near edge" instead
    of an approach-blind nearest-edge guess. Pushes off OTHER (non-anchor)
    footprints keep the original nearest-edge rule (issue #61, unchanged and
    already validated).

    Architectural room-scale AABBs are excluded (issue #53 — not stamped by
    :func:`_synthetic_from_gt`, so not a real local footprint to push off of) and
    elevated instances (base z at/above the terrain slab, i.e. overhangs) never
    blocked the floor to begin with.
    """
    x0, y0, x1, y1 = _gt_footprint_bounds(gt, 0.0)
    room_w, room_h = x1 - x0, y1 - y0
    x, y = xy
    for _ in range(4):  # bounded: converges in one pass for the common single-supporter case
        moved = False
        for rec in gt.instances:
            if float(rec.aabb_min[2]) >= TERRAIN_SLAB_MAX_Z:
                continue  # elevated overhang -- never blocked the floor
            if _is_architectural_room_scale_aabb(rec.aabb_min, rec.aabb_max, room_w, room_h):
                continue  # #53 territory -- not a real local footprint
            axmin = float(rec.aabb_min[0]) - _RUBRIC_GOAL_CLEARANCE_M
            aymin = float(rec.aabb_min[1]) - _RUBRIC_GOAL_CLEARANCE_M
            axmax = float(rec.aabb_max[0]) + _RUBRIC_GOAL_CLEARANCE_M
            aymax = float(rec.aabb_max[1]) + _RUBRIC_GOAL_CLEARANCE_M
            if not (axmin <= x <= axmax and aymin <= y <= aymax):
                continue
            if (
                approach_xy is not None
                and anchor_id is not None
                and rec.instance_id == anchor_id
            ):
                # Issue #66: directional push off the anchor's OWN footprint, biased
                # toward the side the route actually approaches from.
                cx = float(rec.aabb_min[0] + rec.aabb_max[0]) / 2.0
                cy = float(rec.aabb_min[1] + rec.aabb_max[1]) / 2.0
                adx, ady = approach_xy[0] - cx, approach_xy[1] - cy
                if abs(adx) >= abs(ady):
                    x = axmax if adx >= 0 else axmin
                else:
                    y = aymax if ady >= 0 else aymin
            else:
                # Push off an OTHER instance's footprint: nearest edge (issue #61).
                d_left, d_right = x - axmin, axmax - x
                d_bottom, d_top = y - aymin, aymax - y
                m = min(d_left, d_right, d_bottom, d_top)
                if m == d_left:
                    x = axmin
                elif m == d_right:
                    x = axmax
                elif m == d_bottom:
                    y = aymin
                else:
                    y = aymax
            moved = True
        if not moved:
            break
    dist = ((x - xy[0]) ** 2 + (y - xy[1]) ** 2) ** 0.5
    if dist > _RUBRIC_GOAL_MAX_PUSH_M:
        return xy
    return (x, y)


def _if_rubric_geometry(
    text: str,
    gt: GTScene,
    idx: BasicSceneIndex,
    *,
    start_xy: tuple[float, float] | None = None,
) -> tuple[
    list[tuple[str, tuple[float, float]]],
    list[tuple[int, object]],
    list[object],
    list[tuple[int, ...] | None],
    list[tuple[np.ndarray, np.ndarray] | None],
]:
    """Resolve the ordered leg goals, corridor gates, and avoid capsules for the rubric.

    Returns ``(leg_goals, corridor_gates, avoid_capsules, leg_instance_ids,
    leg_instance_aabbs)`` in the GT (object) frame:
      * leg_goals: ``(kind, (x, y))`` per route leg, in order. GOTO/VIA_NEAR -> the
        resolved anchor centroid; CORRIDOR_BETWEEN -> the gate midpoint.
      * corridor_gates: ``(leg_index, Gate)`` per corridor leg (for threading_check).
      * avoid_capsules: one :class:`Capsule` per resolvable AvoidSpec.
      * leg_instance_ids: parallel to ``leg_goals`` — the resolved anchor's
        ``instance_id`` (a 1-tuple for GOTO/VIA_NEAR, a 2-tuple for CORRIDOR_BETWEEN)
        (issue #59 probe: lets the probe report which instance our resolver grounded
        each leg to, independent of the goal xy).
      * leg_instance_aabbs (issue #70): parallel to ``leg_goals`` — the resolved
        anchor's OWN ``(aabb_min, aabb_max)`` for GOTO/VIA_NEAR legs (the SAME
        ``rec`` the goal was projected from, ``rec.aabb_min``/``rec.aabb_max``
        straight off the ``InstanceRecord`` — no extra lookup); ``None`` for
        CORRIDOR_BETWEEN legs (two anchors, no single "referenced instance" to
        widen a pass-by radius around — corridor legs stay STOP-tolerance in
        :func:`core.groundtruth.scoring.score_instruction_rubric`) and for any
        leg whose anchor did not resolve.
    Legs/avoids whose anchors don't resolve are skipped (an unscored, not a wrong, leg).

    ``start_xy`` (issue #66, optional): the point the route departs from — passed
    through to :func:`_nearest_free_goal` as the FIRST leg's approach reference, so
    a GOTO/VIA_NEAR goal that must be pushed off its own anchor's footprint picks
    the side actually approached, not an approach-blind nearest-edge guess. Later
    legs use the PREVIOUS leg's own resolved goal as their approach reference.
    ``None`` (the default) falls back to the old undirected nearest-edge push —
    used by callers (the #59 probe, ``cvsweep``) that don't track a route start.
    """
    from core.parsing.regex_tier import parse_regex
    from core.geometry.toolbox import (
        TargetSpec,
        avoid_capsule,
        corridor_gate,
        resolve,
    )
    from core.geometry import primitives as P
    from core.plan_schema import LegKind, Pred

    #: Superlative disambiguator preds (mirrors core.heads.instruction._SUPERLATIVE_PREDS):
    #: the toolbox already ranks by a real margin for these, so neither correction below
    #: may override it.
    _SUPERLATIVE_PREDS = frozenset({Pred.CLOSEST_TO, Pred.FARTHEST_FROM})

    plan = parse_regex(text)
    leg_goals: list[tuple[str, tuple[float, float]]] = []
    corridor_gates: list[tuple[int, object]] = []
    avoid_capsules: list[object] = []
    leg_instance_ids: list[tuple[int, ...] | None] = []
    leg_instance_aabbs: list[tuple[np.ndarray, np.ndarray] | None] = []
    if not plan.route:
        return leg_goals, corridor_gates, avoid_capsules, leg_instance_ids, leg_instance_aabbs

    def _resolve_anchor_rec(anchor, exclude_id: int | None = None):
        spec = TargetSpec(
            noun=anchor.noun, raw=anchor.raw, attributes=list(anchor.attributes),
            clauses=[anchor.disambiguator] if anchor.disambiguator is not None else [],
        )
        res = resolve(spec, idx)
        ranked = list(res.candidates_ranked)
        disamb = anchor.disambiguator
        has_superlative = disamb is not None and disamb.pred in _SUPERLATIVE_PREDS

        # Issue #71 (RUBRIC-WRONG class): once the resolve fallback ladder has dropped
        # every relation clause (category_only, the noisiest rung -- issue #59), the
        # remaining soft best-effort ranking can rank a loosely/generically matched label
        # ahead of a survivor whose label is an EXACT match for the anchor noun (e.g. a
        # bare "table" outranking an actual "dining table" for a "dining table" query).
        # Prefer an exact-label match within that fallback pool before falling through to
        # the toolbox's own soft order. This is the rubric's OWN independent correction
        # (reads only the anchor spec + this resolve's own audit trail, never the
        # navigation head's state) -- not a substitution of the head's output.
        if not has_superlative and any(r.step == "category_only" for r in res.audit):
            norm = anchor.noun.strip().lower()
            exact = [c for c in ranked if c.label.strip().lower() == norm]
            if exact:
                rest = [c for c in ranked if c.label.strip().lower() != norm]
                ranked = exact + rest

        # Same ordered-route salience InstructionHead._ranked_anchor applies (IF-F3,
        # corrected by issue #71's parity audit to a same-label group only -- reordering
        # the full survivor pool by raw distance let a different-label survivor leapfrog
        # one the toolbox itself ranked ahead on real evidence). Computed independently
        # from the rubric's OWN ``approach_xy`` walk (this leg's departure point, never
        # read from the head), so both sides converge on the correct instance from their
        # own state rather than one copying the other's resolved output.
        if not has_superlative and approach_xy is not None and len(ranked) > 1:
            top_label = ranked[0].label
            same = [c for c in ranked if c.label == top_label]
            rest = [c for c in ranked if c.label != top_label]
            if len(same) > 1:
                ax, ay = approach_xy
                same = sorted(
                    same,
                    key=lambda c: (
                        (float(P._as3(c.centroid)[0]) - ax) ** 2
                        + (float(P._as3(c.centroid)[1]) - ay) ** 2,
                        c.instance_id,
                    ),
                )
            ranked = same + rest

        for c in ranked:
            if exclude_id is None or c.instance_id != exclude_id:
                return c
        return None

    approach_xy = start_xy
    for i, leg in enumerate(plan.route):
        if leg.kind is LegKind.CORRIDOR_BETWEEN and len(leg.anchors) == 2:
            r0 = _resolve_anchor_rec(leg.anchors[0])
            # DISTINCT instances across a corridor leg's two anchors (issue #51):
            # "the two columns" / "between the two X" repeats the same noun for
            # both anchors, and independently resolving each lands on the SAME
            # top-ranked instance -> a zero-width gate the driven path can never
            # cross. Mirrors InstructionHead._resolve_leg_anchors' `used` set so the
            # rubric's scoring gate matches the gate the real planner threads
            # against, instead of scoring a degenerate point no path can cross.
            r1 = _resolve_anchor_rec(
                leg.anchors[1], exclude_id=r0.instance_id if r0 is not None else None
            )
            if r0 is None or r1 is None:
                continue
            gate = corridor_gate(r0, r1, idx)
            mid = (float(gate.midpoint[0]), float(gate.midpoint[1]))
            leg_goals.append(("corridor_between", mid))
            corridor_gates.append((i, gate))
            leg_instance_ids.append((r0.instance_id, r1.instance_id))
            leg_instance_aabbs.append(None)  # two anchors -- no single AABB (issue #70)
            approach_xy = mid
        else:
            rec = _resolve_anchor_rec(leg.anchors[0]) if leg.anchors else None
            if rec is None:
                continue
            c = P._as3(rec.centroid)
            goal_xy = _nearest_free_goal(
                (float(c[0]), float(c[1])), gt,
                anchor_id=rec.instance_id, approach_xy=approach_xy,
            )
            kind = "via_near" if leg.kind is LegKind.VIA_NEAR else "goto"
            leg_goals.append((kind, goal_xy))
            leg_instance_ids.append((rec.instance_id,))
            leg_instance_aabbs.append((rec.aabb_min, rec.aabb_max))
            approach_xy = goal_xy

    for spec in plan.avoid:
        try:
            avoid_capsules.append(avoid_capsule(spec, idx))
        except ValueError:
            continue
    return leg_goals, corridor_gates, avoid_capsules, leg_instance_ids, leg_instance_aabbs


def _min_dist_to_polyline(point: tuple[float, float], traj_xy: np.ndarray) -> float | None:
    """Euclidean distance from ``point`` to the nearest VERTEX of ``traj_xy`` (N, 2).

    Vertex-only (not segment-projected) — matches :func:`gt_leg_ceiling.measure`'s
    ``gt_min_dist_m`` metric so the two probes agree. ``None`` when ``traj_xy`` is
    empty (no reference trajectory to compare against).
    """
    arr = np.asarray(traj_xy, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    p = np.asarray(point, dtype=float)[:2]
    return float(np.min(np.linalg.norm(arr[:, :2] - p, axis=1)))


def _min_dist_driven_to_goal(
    driven_traj: np.ndarray, goal: tuple[float, float]
) -> float | None:
    """Minimum distance from the DRIVEN trajectory (densified, matching the rubric's own
    arrival check) to ``goal`` — the magnitude version of the rubric's boolean "reached"
    (issue #59 probe): lets the probe histogram *how far* a missed leg fell short, not
    just whether it cleared the tolerance.
    """
    from core.groundtruth.scoring import _densify_polyline, ARRIVAL_RESAMPLE_STEP_M

    traj = np.asarray(driven_traj, dtype=float)
    if traj.ndim != 2 or traj.shape[0] == 0:
        return None
    traj = _densify_polyline(traj, ARRIVAL_RESAMPLE_STEP_M)
    g = np.asarray(goal, dtype=float)[:2]
    return float(np.min(np.linalg.norm(traj[:, :2] - g, axis=1)))


def _leg_probe_rows(
    leg_goals: list[tuple[str, tuple[float, float]]],
    leg_instance_ids: list[tuple[int, ...] | None],
    driven: np.ndarray,
    gt_traj_xy: np.ndarray | None,
) -> list[dict]:
    """Per-leg probe diagnostics (issue #59): OUR resolved goal/instance, how far our
    OWN driven path fell short of that goal, and how far our goal sits from the GT
    reference path — split failure into wrong-instance/leg-mismatch (goal far from the
    GT path) vs stop-point/planning offset (goal is near the GT path but our own driven
    trajectory never gets close to it).
    """
    rows: list[dict] = []
    for i, (kind, goal) in enumerate(leg_goals):
        iid = leg_instance_ids[i] if i < len(leg_instance_ids) else None
        dist_driven = _min_dist_driven_to_goal(driven, goal)
        dist_gt_path = (
            _min_dist_to_polyline(goal, gt_traj_xy) if gt_traj_xy is not None else None
        )
        rows.append({
            "i": i,
            "kind": kind,
            "our_goal": [float(goal[0]), float(goal[1])],
            "our_instance_id": list(iid) if iid is not None else None,
            "min_dist_driven_to_goal_m": (
                round(dist_driven, 4) if dist_driven is not None else None
            ),
            "dist_goal_to_gt_traj_m": (
                round(dist_gt_path, 4) if dist_gt_path is not None else None
            ),
        })
    return rows


def _terminal_goal_candidates(
    text: str, idx: BasicSceneIndex, k: int = 8
) -> list[np.ndarray]:
    """Object-frame XY centroids of an IF question's terminal-goal candidates.

    We take the LAST ``GOTO`` route leg's anchor, build a :class:`TargetSpec` from it
    (noun + attributes + disambiguating clause), resolve it on the GT index exactly as
    the instruction head would, and return the top-``k`` candidate XY centroids
    (best-first) — the point the GT ``trajectory_qN.ply`` should end at is one of these.
    Returns ``[]`` when the route has no GOTO leg or the anchor doesn't resolve (so the
    scene fit simply drops that endpoint).

    The ranked list (not just the top pick) is what the scene-level frame fit's fallback
    correspondence search consumes (meth-F11): the resolver's disambiguator can mis-rank
    the terminal object, but the TRUE terminal is still in the candidate set, and the
    rigid endpoint-separation invariant identifies it.
    """
    from core.parsing.regex_tier import parse_regex
    from core.geometry.toolbox import TargetSpec, resolve
    from core.plan_schema import LegKind

    plan = parse_regex(text)
    if not plan.route:
        return []
    goto_legs = [leg for leg in plan.route if leg.kind is LegKind.GOTO and leg.anchors]
    if not goto_legs:
        return []
    anchor = goto_legs[-1].anchors[0]
    spec = TargetSpec(
        noun=anchor.noun,
        raw=anchor.raw,
        attributes=list(anchor.attributes),
        clauses=[anchor.disambiguator] if anchor.disambiguator is not None else [],
    )
    res = resolve(spec, idx)
    return [
        np.asarray(c.centroid, dtype=float).reshape(-1)[:2]
        for c in res.candidates_ranked[: max(1, k)]
    ]


def _terminal_goal_centroid(text: str, idx: BasicSceneIndex) -> np.ndarray | None:
    """Object-frame centroid of an IF question's terminal goal (resolver's TOP pick).

    Thin wrapper over :func:`_terminal_goal_candidates` (k=1) — returns the best
    candidate's XY centroid, or None when nothing resolves.
    """
    cands = _terminal_goal_candidates(text, idx, k=1)
    return cands[0] if cands else None


def _fit_if_frame_over_candidates(
    trajs: list[np.ndarray | None],
    cand_lists: list[list[np.ndarray]],
    *,
    gate_m: float,
) -> tuple[S.Frame2D, float] | None:
    """Fallback scene-level IF frame fit: search terminal-goal candidate combinations.

    The default fit feeds one ``(endpoint -> top-candidate centroid)`` correspondence
    per question; when a resolver mis-ranks a terminal object the two correspondences
    become distance-inconsistent and the rigid fit blows past the gate (meth-F11). Here
    we search over the product of each question's ranked terminal candidates and keep the
    lowest-residual rigid fit. Correspondences stay geometry-anchored — every candidate
    is a real terminal-noun object; only the *pairing* is searched, and the rigid
    endpoint-separation invariant (distance preserved) is what discriminates.

    Returns ``(frame, residual)`` for the best pairing whose residual is within
    ``gate_m`` (the caller only adopts a fit that actually re-aligns the scene), or None
    when fewer than two usable endpoints exist or no pairing clears the gate.
    """
    ends: list[np.ndarray] = []
    cands: list[list[np.ndarray]] = []
    for traj, cl in zip(trajs, cand_lists):
        if traj is None or traj.shape[0] == 0 or not cl:
            continue
        ends.append(np.asarray(traj, dtype=float)[-1, :2])
        cands.append(cl)
    if len(ends) < 2:
        return None  # a single endpoint is translation-only (already gate-passing)

    src = np.asarray(ends, dtype=float)
    best: tuple[S.Frame2D, float] | None = None
    for combo in product(*cands):
        dst = np.asarray(combo, dtype=float)
        frame, residual = S.fit_frame(src, dst)
        if best is None or residual < best[1]:
            best = (frame, residual)
    if best is None or best[1] > gate_m:
        return None
    return best


#: Scenes whose IF frame fit is confirmed UNFITTABLE from the GT data itself (meth-F11),
#: not from our resolution. The two GT trajectory terminal endpoints are mutually
#: inconsistent with any rigid sim->object transform: their sim-frame separation cannot
#: equal the object-frame separation of ANY pairing of the resolved terminal objects, so
#: the two-point rigid residual has a hard floor above the alignment gate — a
#: frame-independent contradiction in the challenge trajectory data. Recorded here so the
#: exclusion reads as a DATA property (documented) rather than a silent friendly-ward drop.
_DATA_UNFITTABLE_IF_SCENES: dict[str, str] = {
    "livingroom_3": (
        "GT q4/q5 terminal endpoints are 1.20 m apart but every pillow x bowl pairing is "
        ">= 3.27 m apart — a frame-independent distance contradiction (rigid two-point "
        "residual floor 1.04 m > 1.0 m gate). The q4 trajectory ends in the dining-chair "
        "corner (1.04 m from a chair, 1.33 m from the nearest pillow), not at any pillow; "
        "no rigid sim->object transform can map the endpoints onto the terminal objects."
    ),
}


# --------------------------------------------------------------------------- records


@dataclass
class GTQuestionScore:
    """One scored ground-truth question."""

    scene: str
    qtype: str
    question: str
    # numerical
    our_count: int | None = None
    gt_count_pipeline: int | None = None
    exact_match: bool | None = None
    gt_count_independent: int | None = None
    independent_source: str = ""
    gt_count_scenegraph: int | None = None
    scenegraph_source: str = ""
    annotated_targets_of_class: int | None = None
    csv_instances_of_class: int | None = None
    #: TRUE numerical yardstick (arch-F3): the human answer from questions.pdf, whether
    #: our count matches it, and the source tag. ``gt_answer_true``/``true_match`` stay
    #: None when no answer key is present or the key's question text fails the guard.
    gt_answer_true: int | None = None
    true_match: bool | None = None
    true_source: str = ""  # "questions_pdf_text" on a guarded match, else ""
    # object_reference
    iou: float | None = None
    gt_target_id: int | None = None
    our_target_id: int | None = None  # instance id our resolver picked (instance-match)
    target_source: str = ""
    match_method: str = ""
    # instruction_following — HEADLINE: rubric proxy over the DRIVEN trajectory (IF-F2)
    rubric_score: float | None = None
    ordered_leg_credit: float | None = None
    n_legs: int | None = None
    n_legs_reached_in_order: int | None = None
    n_threading_violations: int | None = None
    n_avoid_violations: int | None = None
    driven_n_poses: int | None = None
    #: Per-leg rubric geometry + outcomes (meth-F7/F8). ``leg_goals`` is
    #: ``[[kind, [x, y]], ...]`` from :func:`_if_rubric_geometry`; ``leg_outcomes`` is
    #: one dict per leg — ``{"i", "kind", "goal", "reached_in_order", "threaded"}`` —
    #: read straight off the rubric scorer (never recomputed here).
    leg_goals: list | None = None
    leg_outcomes: list | None = None
    #: Issue #59 probe: per-leg diagnostic rows — our resolved instance id, how far our
    #: OWN driven trajectory fell short of our resolved goal (magnitude, not just the
    #: rubric's boolean "reached"), and how far our resolved goal sits from the GT
    #: reference trajectory. See :func:`_leg_probe_rows`. Diagnostic-only: never read by
    #: scoring, does not affect ``rubric_score``/``ordered_leg_credit``.
    leg_probe: list | None = None
    # instruction_following — SECONDARY diagnostics only (never headline)
    frechet_m: float | None = None
    coverage_1m: float | None = None
    our_n_waypoints: int | None = None
    gt_n_waypoints: int | None = None
    frame_aligned: bool | None = None
    fit_residual_m: float | None = None
    note: str = ""
    #: Parse-time notes off the resolved Plan (meth issue #26) — e.g. "unparsed clause
    #: text dropped: ..." — surfaced here so a clause-dropped resolution is reviewable
    #: from the report instead of being parse-time-only trivia. Empty on a clean parse.
    parse_notes: str = ""


# --------------------------------------------------------------------------- per-scene


def _fit_scene_if_frame(
    gt: GTScene,
    idx: BasicSceneIndex,
    if_texts: list[str],
    questions_dir: os.PathLike | str | None,
) -> tuple[
    list[np.ndarray | None],
    list[list[np.ndarray]],
    object | None,
    float | None,
    list[tuple[np.ndarray, np.ndarray]],
]:
    """Fit the scene's sim<->object rigid transform from IF trajectory endpoints.

    Extracted from :func:`score_scene` (issue #70) so a cheap PRE-PASS can compute
    every scene's ``fit_residual_m`` WITHOUT running the expensive simulated drive
    (:func:`_drive_if_trajectory`) that follows it in ``score_scene`` — the residual
    is a pure function of the loaded trajectory PLYs + resolved terminal-goal
    candidates + a rigid least-squares fit, entirely independent of any arrival
    tolerance or driving. This lets a battery run derive its OWN p95 residual across
    scenes (:func:`collect_scene_fit_residuals`) and feed
    :func:`core.groundtruth.arrival.derived_arrival_tol_m` a live measurement
    before the real (expensive) scoring pass, rather than the frozen nominal
    :data:`core.groundtruth.arrival.NOMINAL_FIT_RESIDUAL_P95_M` the head uses.

    Returns ``(if_traj, if_cands, frame, residual, pairs)`` — identical to what
    ``score_scene`` used to compute inline (``pairs`` added so the caller can
    still read the first trajectory's start point for the spawn hint); pure
    refactor, no behaviour change.
    """
    if_traj: list[np.ndarray | None] = []
    if_goal: list[np.ndarray | None] = []
    if_cands: list[list[np.ndarray]] = []
    for i, text in enumerate(if_texts):
        traj_q = _IF_TRAJ_INDEX.get(i)
        traj_arr: np.ndarray | None = None
        if questions_dir is not None and traj_q is not None:
            cand = Path(questions_dir) / gt.scene_name / f"trajectory_q{traj_q}.ply"
            if cand.exists():
                traj_arr = S.load_trajectory_ply(cand)
        if_traj.append(traj_arr)
        cand_list = (
            _terminal_goal_candidates(text, idx) if traj_arr is not None else []
        )
        if_cands.append(cand_list)
        if_goal.append(cand_list[0] if cand_list else None)

    pairs = [
        (t, g) for t, g in zip(if_traj, if_goal) if t is not None and t.shape[0] > 0
    ]
    frame, residual = S.align_scene_trajectories(pairs) if pairs else (None, None)

    # meth-F11 fallback: when the default top-candidate fit fails the alignment gate,
    # a resolver terminal mis-rank is the usual cause — the correct terminal object is
    # still in the ranked candidate set. Search candidate pairings for a rigid fit that
    # clears the gate and adopt it if found. Gate-passing scenes never reach this branch,
    # so the aligned scenes (and their Frechet diagnostics) are left untouched.
    if residual is not None and residual > S._ALIGN_RESIDUAL_GATE_M:
        alt = _fit_if_frame_over_candidates(
            if_traj, if_cands, gate_m=S._ALIGN_RESIDUAL_GATE_M
        )
        if alt is not None:
            frame, residual = alt

    return if_traj, if_cands, frame, residual, pairs


def collect_scene_fit_residuals(
    unity_root: os.PathLike | str,
    *,
    questions_path: os.PathLike | str = DEFAULT_QUESTIONS,
    questions_dir: os.PathLike | str = DEFAULT_QUESTIONS_ROOT,
    scenes: list[str] | None = None,
) -> dict[str, float | None]:
    """Cheap pre-pass (issue #70): one ``fit_residual_m`` per scene, no driving.

    Used to compute a battery run's LIVE p95 fit residual before the real scoring
    pass, so :func:`core.groundtruth.arrival.derived_arrival_tol_m` is fed a
    measurement from THIS run's own scenes rather than the frozen nominal snapshot
    the navigation head uses. ``None`` for a scene with no fittable IF trajectory
    (excluded from the p95 by the caller, same as any other missing measurement).
    """
    root = Path(unity_root)
    with open(questions_path, encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[str, float | None] = {}
    for entry in data:
        scene_name = entry["scene"]
        if scenes is not None and scene_name not in scenes:
            continue
        folder = _find_scene_folder(root, scene_name)
        if folder is None:
            continue
        gt = load_scene(folder, scene_name=scene_name)
        idx = BasicSceneIndex(gt.instances)
        if_texts = entry["questions"].get("instruction_following", [])
        _traj, _cands, _frame, residual, _pairs = _fit_scene_if_frame(
            gt, idx, if_texts, questions_dir
        )
        out[scene_name] = residual
    return out


def score_scene(
    gt: GTScene,
    questions: dict[str, list[str]],
    *,
    referential: dict | None = None,
    scene_graph: dict | None = None,
    questions_dir: os.PathLike | str | None = None,
    answers: dict | None = None,
    drive_if: bool = True,
    no_spawn_hint: bool = False,
    walls: bool = True,
    unity_scenes_ros2_root: os.PathLike | str | None = None,
    tol: float | None = None,
) -> list[GTQuestionScore]:
    """Score every question of one GT scene.

    ``tol`` (issue #70): the leg-arrival tolerance passed to
    :func:`core.groundtruth.scoring.score_instruction_rubric`. ``None`` (the
    default) falls back to that function's own default
    (:data:`core.groundtruth.scoring.LEG_ARRIVAL_TOL_M`, the NOMINAL derived
    value). A caller running a full battery should instead derive a LIVE value
    from this run's own scenes' fit residuals
    (:func:`collect_scene_fit_residuals` + p95 +
    :func:`core.groundtruth.arrival.derived_arrival_tol_m`) and pass it here —
    see :func:`run_gt_battery`'s ``tol`` parameter.

    ``no_spawn_hint`` (IF-F2 realism knob): when True, the IF planner spawns at the
    scene centroid instead of the GT trajectory's mapped start, so the exploration cost
    of *finding* the route from a neutral start is visible (eval never hands us the GT
    start).

    ``walls`` (IF-F2 wall realism, on by default): the mirror costmap otherwise has
    object obstacles + an outer boundary but no interior walls, so planned routes can
    cut through where real walls would block them. When a per-scene sim->object frame
    fit is available (below) we derive interior wall cells from the scene's
    ``traversable_area.ply`` (see :func:`_scene_wall_cells`) and stamp them into the
    mirror costmap for every IF question in the scene. ``walls=False`` reproduces the
    old boundary-only costmap unconditionally (the ``--no-walls`` CLI escape).
    """
    idx = BasicSceneIndex(gt.instances)
    out: list[GTQuestionScore] = []

    for text in questions.get("numerical", []):
        ns = S.score_numerical(text, idx, referential=referential, scene_graph=scene_graph)
        gt_true, true_match, true_source, key_note = _true_numerical(
            answers, gt.scene_name, text, ns.our_count
        )
        note = "; ".join(filter(None, [ns.note, key_note]))
        out.append(
            GTQuestionScore(
                scene=gt.scene_name, qtype=QType.NUMERICAL.value, question=text,
                our_count=ns.our_count, gt_count_pipeline=ns.gt_count_pipeline,
                exact_match=ns.exact_match, gt_count_independent=ns.gt_count_independent,
                independent_source=ns.independent_source,
                gt_count_scenegraph=ns.gt_count_scenegraph,
                scenegraph_source=ns.scenegraph_source,
                annotated_targets_of_class=ns.annotated_targets_of_class,
                csv_instances_of_class=ns.csv_instances_of_class,
                gt_answer_true=gt_true, true_match=true_match, true_source=true_source,
                note=note, parse_notes=ns.parse_notes,
            )
        )

    for text in questions.get("object_reference", []):
        ors = S.score_object_reference(text, idx, gt.instances, referential=referential)
        out.append(
            GTQuestionScore(
                scene=gt.scene_name, qtype=QType.OBJECT_REFERENCE.value, question=text,
                iou=(None if ors.iou != ors.iou else round(ors.iou, 4)),
                gt_target_id=ors.gt_target_id, our_target_id=ors.our_target_id,
                target_source=ors.target_source,
                match_method=ors.match_method, note=ors.note,
                parse_notes=ors.parse_notes,
            )
        )

    # Instruction following: two passes. First resolve each question's terminal goal +
    # load its GT trajectory; fit ONE scene-level sim->object transform from the
    # endpoints (+ shared start); then score each with the fitted frame applied.
    if_texts = questions.get("instruction_following", [])
    if_traj, if_cands, frame, residual, pairs = _fit_scene_if_frame(
        gt, idx, if_texts, questions_dir
    )

    # The GT trajectory's (shared) start, mapped into the object frame, is the robot
    # spawn our planner should depart from — feed it so our path and the GT path start
    # at the same place (fair comparison). Fall back to None (scene-corner) with no fit.
    spawn_xy: tuple[float, float] | None = None
    if no_spawn_hint:
        # Realism knob (IF-F2): ignore the GT-matched start; spawn at the scene centroid
        # so the exploration cost of reaching the route from a neutral pose is visible.
        mins = np.array([r.aabb_min for r in gt.instances])
        maxs = np.array([r.aabb_max for r in gt.instances])
        spawn_xy = (
            float((mins[:, 0].min() + maxs[:, 0].max()) / 2),
            float((mins[:, 1].min() + maxs[:, 1].max()) / 2),
        )
    elif frame is not None and pairs:
        start_pt = pairs[0][0][0, :2]
        mapped = frame.apply(np.asarray([start_pt], dtype=float))[0]
        spawn_xy = (float(mapped[0]), float(mapped[1]))

    # IF-F2 wall realism: derive interior wall cells once per scene (all IF questions in
    # the scene share the mirror costmap) from the fitted sim->object frame above. None
    # when disabled, no frame was fit, or the scene ships no traversable mesh — the
    # planner then sees the old boundary-only costmap.
    #
    # Wall derivation uses the TIGHTER WALL_FIT_MAX_RESIDUAL_M gate (issue #52), not the
    # looser scoring alignment gate (S._ALIGN_RESIDUAL_GATE_M) — a fit marginal enough to
    # be untrustworthy for hard wall placement (misprojects the traversable mesh, can wall
    # off real floor) can still be fine for the smooth Frechet/coverage diagnostics.
    frame_for_walls = (
        frame if residual is None or residual <= WALL_FIT_MAX_RESIDUAL_M else None
    )
    wall_cells = (
        _scene_wall_cells(gt, frame_for_walls, unity_scenes_ros2_root=unity_scenes_ros2_root)
        if walls
        else None
    )

    from core.parsing.regex_tier import parse_regex as _parse_regex_if

    for i, text in enumerate(if_texts):
        traj_q = _IF_TRAJ_INDEX.get(i)
        rec = GTQuestionScore(
            scene=gt.scene_name, qtype=QType.INSTRUCTION_FOLLOWING.value, question=text,
            parse_notes=_parse_regex_if(text).notes,
        )
        traj_path = None
        if questions_dir is not None and traj_q is not None:
            cand = Path(questions_dir) / gt.scene_name / f"trajectory_q{traj_q}.ply"
            if cand.exists():
                traj_path = cand
        if traj_path is None:
            rec.note = "no GT trajectory file found; IF unscored"
            out.append(rec)
            continue

        if not drive_if:
            rec.note = "drive_if disabled; IF unscored"
            out.append(rec)
            continue

        # HEADLINE: rubric proxy over the DRIVEN trajectory (IF-F2). We simulate the
        # drive (constant-speed kinematic follower over the planned breadcrumbs), then
        # score ordered per-leg arrival + threading + avoid violations. The planned-path
        # Frechet/coverage are carried through the rubric as SECONDARY diagnostics only.
        driven = _drive_if_trajectory(
            text, gt, idx, start_xy=spawn_xy, wall_cells=wall_cells
        )
        leg_goals, corridor_gates, avoid_caps, leg_instance_ids, leg_instance_aabbs = (
            _if_rubric_geometry(text, gt, idx, start_xy=spawn_xy)
        )
        rub = S.score_instruction_rubric(
            driven,
            leg_goals,
            corridor_gates=corridor_gates,
            avoid_capsules=avoid_caps,
            trajectory_ply=traj_path,
            frame=frame,
            leg_instance_aabbs=leg_instance_aabbs,
            **({"tol": tol} if tol is not None else {}),
        )
        rec.rubric_score = round(rub.rubric_score, 4)
        rec.ordered_leg_credit = round(rub.ordered_leg_credit, 4)
        rec.n_legs = rub.n_legs
        rec.n_legs_reached_in_order = rub.n_legs_reached_in_order
        rec.n_threading_violations = rub.n_threading_violations
        rec.n_avoid_violations = rub.n_avoid_violations
        rec.driven_n_poses = rub.driven_n_poses
        # Per-leg rubric geometry + outcomes into the row (meth-F7/F8): the resolved
        # ordered leg goals and the scorer's per-leg arrival/threading verdicts, read
        # straight off ``rub`` (not recomputed) so results carry per-leg provenance.
        rec.leg_goals = [
            [kind, [float(gx), float(gy)]] for kind, (gx, gy) in leg_goals
        ]
        rec.leg_outcomes = [
            {
                "i": o.index,
                "kind": o.kind,
                "goal": [float(o.goal_xy[0]), float(o.goal_xy[1])],
                "reached_in_order": bool(o.reached_in_order),
                "threaded": o.threaded,
                # issue #70: which arrival semantics + tolerance actually applied.
                "pass_by": bool(o.pass_by),
                "tol_used": round(float(o.tol_used), 4),
            }
            for o in rub.leg_outcomes
        ]
        # Issue #59 probe: per-leg goal/instance + distance diagnostics, computed
        # independently of the scorer (no scoring semantics touched) so a failing leg's
        # failure mode is directly readable off the row.
        traj_arr_i = if_traj[i]
        gt_traj_xy = None
        if traj_arr_i is not None and traj_arr_i.shape[0] > 0:
            gt_traj_xy = frame.apply(traj_arr_i) if frame is not None else traj_arr_i[:, :2]
        rec.leg_probe = _leg_probe_rows(leg_goals, leg_instance_ids, driven, gt_traj_xy)
        # our_n_waypoints mirrors the driven pose count for the IF rubric path (the
        # trajectory we scored). It stayed None after the wave rebuilt IF scoring onto
        # the rubric proxy, which broke test_score_scene_if_produces_two_numbers — a
        # pre-existing gate failure independent of T11; set it so the diagnostic pair
        # (our vs GT waypoint count) is populated again.
        rec.our_n_waypoints = rub.driven_n_poses
        # secondary diagnostics (frame alignment + planned-path frechet/coverage)
        rec.frechet_m = rub.frechet_m
        rec.coverage_1m = round(rub.coverage_1m, 4) if rub.coverage_1m is not None else None
        rec.gt_n_waypoints = int(S.load_trajectory_ply(traj_path).shape[0])
        rec.frame_aligned = residual is None or residual <= S._ALIGN_RESIDUAL_GATE_M
        rec.fit_residual_m = round(residual, 4) if residual is not None else None
        # meth-F11: a scene that stays unaligned AND is a confirmed GT-data defect carries
        # the data-confirmed reason on the row, so the exclusion reads as a documented
        # data property rather than a silent (friendly-ward) drop.
        data_note = ""
        if rec.frame_aligned is False and gt.scene_name in _DATA_UNFITTABLE_IF_SCENES:
            data_note = (
                "frame fit unaligned — DATA-CONFIRMED unfittable (meth-F11): "
                + _DATA_UNFITTABLE_IF_SCENES[gt.scene_name]
            )
        # IF-F2 wall realism: note when walls were requested but unavailable, so a
        # scene's costmap fidelity is visible from the report (not a silent fallback).
        wall_note = ""
        if walls and wall_cells is None:
            if frame is None:
                reason = "no fitted sim->object frame"
            elif frame_for_walls is None:
                reason = (
                    f"frame fit residual {residual:.2f} m > "
                    f"{WALL_FIT_MAX_RESIDUAL_M:.1f} m wall-derivation gate"
                )
            else:
                reason = "no traversable_area.ply for scene"
            wall_note = f"interior walls unavailable ({reason}); boundary-only costmap"
        detail = "; ".join(
            filter(
                None,
                [rub.note] + rub.threading_details + rub.avoid_details
                + [data_note, wall_note],
            )
        )
        rec.note = detail
        out.append(rec)

    return out


# --------------------------------------------------------------------------- discovery


def _find_scene_folder(unity_root: Path, scene_name: str) -> Path | None:
    """Locate a scene's folder under the Unity root (verbatim name match)."""
    direct = unity_root / scene_name
    if direct.is_dir():
        return direct
    # Some zips nest under a 'Unity/' dir; probe one level down.
    nested = unity_root / "Unity" / scene_name
    if nested.is_dir():
        return nested
    for child in unity_root.iterdir() if unity_root.is_dir() else []:
        if child.is_dir() and (child / f"{scene_name}_object_result.csv").exists():
            return child
        cand = child / scene_name
        if cand.is_dir():
            return cand
    return None


def _load_answers(answers_path: os.PathLike | str | None) -> dict | None:
    """Load the true numerical answer key (arch-F3); return None when absent/unreadable.

    A *missing* key file is a soft condition — the battery still runs and every true
    field stays null. A *present-but-unreadable* key (corrupt/unparseable) is NOT soft:
    it silently vanishes the TRUE-accuracy yardstick, which is the exact failure the
    yardstick exists to prevent, so we emit a loud stderr warning (distinct from the
    silent missing case) before degrading to null. Either way we never abort the run —
    see :func:`_answer_key_status` for the classification surfaced in the report topline.
    """
    if answers_path is None:
        return None
    p = Path(answers_path)
    if not p.exists():
        return None
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"gt_battery: WARNING answer key present but UNREADABLE at {p} "
            f"({exc.__class__.__name__}: {exc}) — TRUE-accuracy yardstick unavailable; "
            "this is a corrupt key, NOT a legitimately absent one",
            file=sys.stderr,
        )
        return None


def _answer_key_status(answers_path: os.PathLike | str | None) -> str:
    """Classify the answer key for the report topline: ``"ok"`` | ``"missing"`` | ``"unreadable"``.

    ``"missing"`` = no file (soft, expected before the key is extracted); ``"unreadable"``
    = the file is present but corrupt/unparseable (the yardstick has silently vanished —
    surfaced loudly in the topline so it can't pass for a legitimately absent key). Pure
    classification only: the loud stderr warning lives in :func:`_load_answers`.
    """
    if answers_path is None:
        return "missing"
    p = Path(answers_path)
    if not p.exists():
        return "missing"
    try:
        with open(p, encoding="utf-8") as fh:
            json.load(fh)
    except (OSError, json.JSONDecodeError):
        return "unreadable"
    return "ok"


def _squash_ws(s: str) -> str:
    """Whitespace-stripped, casefolded form for guard comparison.

    The answer-key ``question_raw`` lost its spaces during PDF text extraction, so the
    guard compares question text with ALL whitespace removed and case folded.
    """
    return "".join(str(s).split()).casefold()


def _true_numerical(
    answers: dict | None,
    scene_name: str,
    question_text: str,
    our_count: int | None,
) -> tuple[int | None, bool | None, str, str]:
    """Resolve the true numerical answer for one question against the answer key.

    Returns ``(gt_answer_true, true_match, true_source, note)``. Matching is by scene
    (exactly one numerical question per scene) and then GUARDED: the key's
    ``question_raw`` must equal the battery question text under
    :func:`_squash_ws`. A mismatch never mis-anchors — it returns nulls plus the note
    ``"answer-key question mismatch"``. A scene absent from the key returns nulls with
    no note (soft miss); no key at all returns nulls with no note.
    """
    if not answers:
        return None, None, "", ""
    entry = (answers.get("scenes") or {}).get(scene_name)
    if not isinstance(entry, dict):
        return None, None, "", ""
    if _squash_ws(entry.get("question_raw", "")) != _squash_ws(question_text):
        return None, None, "", "answer-key question mismatch"
    try:
        answer = int(entry["answer"])
    except (KeyError, TypeError, ValueError):
        return None, None, "", "answer-key answer unreadable"
    match = None if our_count is None else (our_count == answer)
    return answer, match, "questions_pdf_text", ""


def _load_referential(folder: Path, scene_name: str) -> dict | None:
    p = folder / f"{scene_name}_referential_statements.json"
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _load_scene_graph(folder: Path, scene_name: str) -> dict | None:
    p = folder / f"{scene_name}_scene_graph.json"
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def run_gt_battery(
    unity_root: os.PathLike | str,
    *,
    questions_path: os.PathLike | str = DEFAULT_QUESTIONS,
    questions_dir: os.PathLike | str = DEFAULT_QUESTIONS_ROOT,
    answers_path: os.PathLike | str | None = DEFAULT_ANSWERS,
    scenes: list[str] | None = None,
    drive_if: bool = True,
    no_spawn_hint: bool = False,
    walls: bool = True,
    unity_scenes_ros2_root: os.PathLike | str | None = None,
    tol: float | None = None,
    derive_tol: bool = False,
) -> tuple[list[GTQuestionScore], list[str]]:
    """Score every question whose scene folder is present under ``unity_root``.

    ``tol`` (issue #70): explicit leg-arrival tolerance, forwarded to every
    :func:`score_scene` call. ``derive_tol`` (issue #70, mutually exclusive with
    an explicit ``tol``): run the cheap :func:`collect_scene_fit_residuals`
    pre-pass first (no driving), compute THIS run's own p95 fit residual across
    its scenes, and derive ``tol`` from it
    (:func:`core.groundtruth.arrival.derived_arrival_tol_m`) before the real
    (expensive) scoring pass -- so the tolerance reflects THIS run's own
    measured frame-fit uncertainty rather than the frozen nominal snapshot
    :data:`core.groundtruth.scoring.LEG_ARRIVAL_TOL_M` falls back to. Both
    ``None``/``False`` (the default) leaves every ``score_scene`` call at ITS
    own default (the nominal constant) -- unchanged pre-#70 behaviour.

    Returns (scores, missing_scenes).
    """
    if derive_tol:
        if tol is not None:
            raise ValueError("pass either tol= or derive_tol=True, not both")
        residuals = collect_scene_fit_residuals(
            unity_root,
            questions_path=questions_path,
            questions_dir=questions_dir,
            scenes=scenes,
        )
        finite = [r for r in residuals.values() if r is not None]
        if finite:
            p95 = float(np.percentile(finite, 95))
            tol = S.derived_arrival_tol_m(p95)

    root = Path(unity_root)
    with open(questions_path, encoding="utf-8") as fh:
        data = json.load(fh)
    answers = _load_answers(answers_path)

    scores: list[GTQuestionScore] = []
    missing: list[str] = []
    for entry in data:
        scene_name = entry["scene"]
        if scenes is not None and scene_name not in scenes:
            continue
        folder = _find_scene_folder(root, scene_name)
        if folder is None:
            missing.append(scene_name)
            continue
        gt = load_scene(folder, scene_name=scene_name)
        referential = _load_referential(folder, scene_name)
        scene_graph = _load_scene_graph(folder, scene_name)
        scores.extend(
            score_scene(
                gt,
                entry["questions"],
                referential=referential,
                scene_graph=scene_graph,
                questions_dir=questions_dir,
                answers=answers,
                drive_if=drive_if,
                no_spawn_hint=no_spawn_hint,
                walls=walls,
                unity_scenes_ros2_root=unity_scenes_ros2_root,
                tol=tol,
            )
        )
    return scores, missing


# --------------------------------------------------------------------------- aggregate


def aggregate(scores: list[GTQuestionScore]) -> dict:
    """Per-type accuracy topline."""
    num = [s for s in scores if s.qtype == QType.NUMERICAL.value]
    obj = [s for s in scores if s.qtype == QType.OBJECT_REFERENCE.value]
    inf = [s for s in scores if s.qtype == QType.INSTRUCTION_FOLLOWING.value]

    def _mean(vals: list[float]) -> float | None:
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    num_exact = [1.0 if s.exact_match else 0.0 for s in num]
    # TRUE accuracy (arch-F3): mean of true_match over rows that HAVE a true answer.
    num_true_rows = [s for s in num if s.gt_answer_true is not None]
    num_true = [1.0 if s.true_match else 0.0 for s in num_true_rows]
    # NUM-F6(b): a ``*_class_only`` count is relation-agnostic, NOT independent evidence
    # for the relation-filtered question — exclude those rows from the agreement stat
    # (report them separately as "no independent evidence"), leaving only strict
    # ``referential`` / ``scene_graph`` rows.
    num_evidence = [s for s in num if s.independent_source == "referential"]
    num_agree = [
        1.0 if s.our_count == s.gt_count_independent else 0.0 for s in num_evidence
    ]
    num_sg_evidence = [s for s in num if s.scenegraph_source == "scene_graph"]
    num_sg_agree = [
        1.0 if s.our_count == s.gt_count_scenegraph else 0.0 for s in num_sg_evidence
    ]
    num_no_evidence = sum(
        1
        for s in num
        if s.independent_source != "referential" and s.scenegraph_source != "scene_graph"
    )
    obj_iou = [s.iou for s in obj if s.iou is not None]
    obj_scored = [s for s in obj if s.iou is not None]
    # match-method breakdown across ALL object-reference questions
    method_counts: dict[str, int] = {}
    for s in obj:
        m = s.match_method or "none"
        method_counts[m] = method_counts.get(m, 0) + 1

    # Instruction following (IF-F2): HEADLINE is the rubric-proxy score over the driven
    # trajectory; Frechet/coverage are secondary diagnostics only. Aligned vs unaligned
    # is retained for the diagnostic columns.
    inf_scored = [s for s in inf if s.rubric_score is not None]
    inf_aligned = [s for s in inf if s.frame_aligned]
    unaligned_scenes = sorted({s.scene for s in inf if s.frame_aligned is False})
    # meth-F11: partition the unaligned set into DATA-confirmed unfittable scenes (the GT
    # trajectory endpoints cannot be rigidly mapped to any terminal-object pairing — a
    # documented data defect) vs any residual unexplained gap. With the candidate-search
    # fallback in place the only unaligned scenes should be data-confirmed; a scene
    # appearing in ``unaligned_scenes_unexplained`` is a genuine resolution regression.
    unaligned_data_confirmed = [
        s for s in unaligned_scenes if s in _DATA_UNFITTABLE_IF_SCENES
    ]
    unaligned_unexplained = [
        s for s in unaligned_scenes if s not in _DATA_UNFITTABLE_IF_SCENES
    ]
    return {
        "numerical": {
            "n": len(num),
            # TRUE accuracy is the primary yardstick; determinism is a secondary signal.
            "n_with_true_answer": len(num_true_rows),
            "true_accuracy": _mean(num_true) if num_true else None,
            # Renamed from ``exact_match_rate_pipeline``: it measures pipeline
            # determinism (our count == our count over GT geometry), NOT accuracy.
            "pipeline_determinism_rate": _mean(num_exact),
            "n_with_independent": len(num_agree),
            "independent_agreement_rate": _mean(num_agree) if num_agree else None,
            "n_with_scenegraph": len(num_sg_agree),
            "scenegraph_agreement_rate": _mean(num_sg_agree) if num_sg_agree else None,
            "n_no_independent_evidence": num_no_evidence,
        },
        "object_reference": {
            "n": len(obj),
            "n_scored": len(obj_scored),
            "mean_iou": _mean(obj_iou),
            "iou_at_0p25": _mean([1.0 if v >= 0.25 else 0.0 for v in obj_iou]) if obj_iou else None,
            "iou_at_0p5": _mean([1.0 if v >= 0.5 else 0.0 for v in obj_iou]) if obj_iou else None,
            "match_method_breakdown": method_counts,
        },
        "instruction_following": {
            "n": len(inf),
            "n_scored": len(inf_scored),
            # HEADLINE (rubric proxy over driven trajectory)
            "mean_rubric_score": _mean([s.rubric_score for s in inf_scored]),
            "mean_ordered_leg_credit": _mean([s.ordered_leg_credit for s in inf_scored]),
            "total_threading_violations": sum(
                s.n_threading_violations or 0 for s in inf_scored
            ),
            "total_avoid_violations": sum(
                s.n_avoid_violations or 0 for s in inf_scored
            ),
            # SECONDARY diagnostics only (frame-aligned planned-path shape metrics)
            "n_aligned": len(inf_aligned),
            "n_unaligned_scenes": len(unaligned_scenes),
            "unaligned_scenes": unaligned_scenes,
            # meth-F11: exclusions split by cause. ``data_confirmed`` scenes are a
            # documented GT-data defect (endpoints not rigidly mappable), NOT a
            # friendly-ward drop; ``unexplained`` should be empty.
            "unaligned_scenes_data_confirmed": unaligned_data_confirmed,
            "unaligned_scenes_unexplained": unaligned_unexplained,
            "mean_frechet_m_aligned_diag": _mean([s.frechet_m for s in inf_aligned]),
            "mean_coverage_1m_aligned_diag": _mean([s.coverage_1m for s in inf_aligned]),
        },
    }


# --------------------------------------------------------------------------- report


def _md_table(scores: list[GTQuestionScore]) -> str:
    header = (
        "| Scene | Type | Metric(s) | Note | Question |\n"
        "|---|---|---|---|---|\n"
    )
    rows = []
    for s in scores:
        q = s.question if len(s.question) <= 55 else s.question[:52] + "..."
        if s.qtype == QType.NUMERICAL.value:
            indep = (
                f", indep={s.gt_count_independent}({s.independent_source})"
                if s.gt_count_independent is not None
                else ""
            )
            sg = (
                f", sg={s.gt_count_scenegraph}({s.scenegraph_source})"
                if s.gt_count_scenegraph is not None
                else ""
            )
            cov_ann = (
                f", ann_cov={s.annotated_targets_of_class}/{s.csv_instances_of_class}"
                if s.csv_instances_of_class is not None
                else ""
            )
            metric = (
                f"count={s.our_count} pipeline_gt={s.gt_count_pipeline}{indep}{sg}{cov_ann}"
            )
        elif s.qtype == QType.OBJECT_REFERENCE.value:
            iou = "n/a" if s.iou is None else f"{s.iou:.3f}"
            metric = (
                f"IoU={iou} tgt={s.gt_target_id}({s.target_source}"
                f"/{s.match_method or 'none'})"
            )
        else:
            # HEADLINE: rubric proxy over the driven trajectory; frechet/coverage secondary.
            if s.rubric_score is None:
                metric = "IF unscored"
            else:
                fr = "n/a" if s.frechet_m is None else f"{s.frechet_m:.2f}m"
                cov = "n/a" if s.coverage_1m is None else f"{s.coverage_1m:.0%}"
                metric = (
                    f"rubric={s.rubric_score:.2f} "
                    f"legs={s.n_legs_reached_in_order}/{s.n_legs} "
                    f"thread_viol={s.n_threading_violations} "
                    f"avoid_viol={s.n_avoid_violations} poses={s.driven_n_poses} "
                    f"| diag: Frechet={fr} cover1m={cov}"
                )
        note = s.note or ""
        rows.append(f"| {s.scene} | {s.qtype[:4]} | {metric} | {note} | {q} |")
    return header + "\n".join(rows) + "\n"


def _or_instance_match(scores: list[GTQuestionScore]) -> tuple[list[GTQuestionScore], int]:
    """Object-reference instance-match count (arch-F3 headline; IoU is diagnostic).

    Returns (scoreable_rows, n_instance_matched), over OR questions where a GT
    target was matched (scoreable).
    """
    obj_rows = [s for s in scores if s.qtype == QType.OBJECT_REFERENCE.value]
    or_scored = [s for s in obj_rows if s.gt_target_id is not None]
    or_instance_match = sum(
        1
        for s in or_scored
        if s.our_target_id is not None and s.our_target_id == s.gt_target_id
    )
    return or_scored, or_instance_match


def write_report(
    scores: list[GTQuestionScore],
    missing: list[str],
    out_dir: os.PathLike | str,
    *,
    argv: list[str] | None = None,
    answer_key_status: str | None = None,
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    agg = aggregate(scores)

    md_path = out / "gt_battery_report.md"
    json_path = out / "gt_battery_results.json"

    scenes = sorted({s.scene for s in scores})
    lines: list[str] = []
    lines.append(f"# GT battery — REAL accuracy report ({date.today().isoformat()})\n")
    lines.append(
        f"{len(scores)} question(s) across {len(scenes)} ground-truth scene(s): "
        f"{', '.join(scenes) or '(none)'}. "
        f"Missing/skipped scenes: {', '.join(missing) or 'none'}.\n"
    )
    lines.append(
        "> **Yardstick note (numerical):** the PRIMARY yardstick is now `TRUE accuracy` "
        "— our count vs the human answer extracted from each scene's questions.pdf "
        "(`gt_answer_true`, source `questions_pdf_text`), matched by scene under a "
        "whitespace-insensitive question-text guard (a mismatch is left null, never "
        "mis-anchored). The `pipeline_gt` exact-match is DEMOTED to a self-consistency "
        "signal (our resolver over GT geometry — measures determinism, not truth). The "
        "`indep` column is a second opinion from the referential-statement annotations "
        "(distinct annotated target instances); `referential_class_only` means the "
        "count is relation-agnostic (coarser). Disagreements are the informative "
        "signal.\n"
    )
    lines.append(
        "> **OBB->AABB note:** GT boxes are the axis-aligned hull of each object's "
        "oriented box (rotated corners, min/max), a strict over-approximation for "
        "non-axis-aligned objects — small IoU deficits on rotated targets are partly "
        "this, not localisation error.\n"
    )
    lines.append(
        "> **IF headline is the rubric proxy (IF-F2):** the instruction-following "
        "headline is `rubric` — ordered per-leg arrival credit over the DRIVEN "
        "trajectory (a constant-speed kinematic follower over the planned breadcrumbs; "
        "a v1 simplification, no local-planner deviation / waypoint-snapping), minus one "
        "leg-equivalent penalty per corridor leg never threaded (`threading_check`) and "
        "per avoid capsule breached (`capsule_violated`). `legs=k/n` is ordered legs "
        "reached. Fréchet/coverage after `| diag:` are SECONDARY planned-path shape "
        "diagnostics only — never the headline (they measure shape similarity to the "
        "reference PLY, which the challenge rubric does not score).\n"
    )
    walls_disabled = argv is not None and "--no-walls" in argv
    if walls_disabled:
        lines.append(
            "> **Wall realism (IF-F2) — disabled this run (`--no-walls`):** the mirror "
            "costmap used for IF planning has object obstacles + an outer boundary but NO "
            "interior walls, so planned paths may cut through where real walls are. Read "
            "cross-room routes with that caveat. Interior walls CAN be derived from each "
            "scene's `traversable_area.ply` (see the default-on note below) — this run "
            "opted out.\n"
        )
    else:
        lines.append(
            "> **Wall realism (IF-F2) — interior walls derived from `traversable_area.ply` "
            "(default on):** the VLA-3D region data ships only per-region AABBs (no "
            "door/passage geometry), so those can't source real walls — but each scene's "
            "`traversable_area.ply` (a floor-traversability point mesh in the sim/"
            "trajectory frame) can. Its coverage is rasterized to the mirror costmap's grid "
            "via the scene's fitted sim->object frame; grid cells inside the scene's outer "
            "boundary that the mesh does NOT cover (after a small dilation to bridge "
            "point-cloud sampling gaps) become OBSTACLE. A per-question row's note flags "
            "`interior walls unavailable` when the scene had no fitted frame or no "
            "`traversable_area.ply` to derive walls from — those fall back to the old "
            "boundary-only costmap. `--no-walls` reproduces the old behaviour "
            "unconditionally.\n"
        )

    n, o, i = agg["numerical"], agg["object_reference"], agg["instruction_following"]
    method_str = ", ".join(
        f"{k}={v}" for k, v in sorted(o["match_method_breakdown"].items())
    ) or "none"

    # TRUE-accuracy leader (arch-F3): k correct / n questions that had a true answer.
    num_rows = [s for s in scores if s.qtype == QType.NUMERICAL.value]
    n_true = n["n_with_true_answer"]
    k_true = sum(1 for s in num_rows if s.true_match)
    if n_true:
        true_lead = f"TRUE accuracy {k_true}/{n_true} (answer key: questions.pdf)"
    elif answer_key_status == "unreadable":
        # A corrupt/unparseable key silently degrades to zero keyed rows — same shape as
        # a legitimately absent key. Say so loudly here so the yardstick can't vanish
        # unnoticed (issue #16).
        true_lead = (
            "true accuracy n/a (answer key present but UNREADABLE — corrupt/unparseable)"
        )
    else:
        true_lead = "true accuracy n/a (no answer key)"
    # Instance-match (arch-F3): resolved instance id == gt_target_id, over the OR
    # questions where a GT target was matched (scoreable). Real-perception IoU pending.
    or_scored, or_instance_match = _or_instance_match(scores)

    lines.append("## Topline (per type)\n")
    lines.append(
        f"- **Numerical** (n={n['n']}): {true_lead}; independent (referential) agreement "
        f"{_pct(n['independent_agreement_rate'])} over {n['n_with_independent']} with "
        f"strict evidence; scene-graph agreement {_pct(n['scenegraph_agreement_rate'])} "
        f"over {n['n_with_scenegraph']}; {n['n_no_independent_evidence']} question(s) had "
        f"no independent evidence (class-only counts excluded from agreement).\n"
        f"- **Object reference** (n={o['n']}): instance-match {or_instance_match}/"
        f"{len(or_scored)} scored (scoreability {len(or_scored)}/{o['n']}); IoU pending "
        f"real perception. Match method: {method_str}.\n"
        f"- **Instruction following** (n={i['n']}, scored={i['n_scored']}): HEADLINE "
        f"mean rubric-proxy score {_num(i['mean_rubric_score'])} (mean ordered-leg "
        f"credit {_num(i['mean_ordered_leg_credit'])}; {i['total_threading_violations']} "
        f"threading + {i['total_avoid_violations']} avoid violation(s) total). SECONDARY "
        f"(diagnostic only): aligned planned-path mean Frechet "
        f"{_num(i['mean_frechet_m_aligned_diag'])} m, coverage-1m "
        f"{_pct(i['mean_coverage_1m_aligned_diag'])} over {i['n_aligned']} aligned; "
        f"{i['n_unaligned_scenes']} scene(s) unaligned"
        f"{': ' + ', '.join(i['unaligned_scenes']) if i['unaligned_scenes'] else ''}.\n"
    )
    lines.append("\n## Per-question\n")
    lines.append(_md_table(scores))

    # Parse diagnostics (issue #26): parse-time notes (e.g. "unparsed clause text
    # dropped: ...") are otherwise invisible outside a one-off debug run — surface any
    # non-empty ones here so a clause-dropped resolution is reviewable from the report.
    parse_diag_rows = [s for s in scores if s.parse_notes]
    if parse_diag_rows:
        lines.append("\n## Parse diagnostics\n")
        lines.append("| Scene | Type | Question | Parse notes |\n|---|---|---|---|\n")
        for s in parse_diag_rows:
            q = s.question if len(s.question) <= 55 else s.question[:52] + "..."
            lines.append(f"| {s.scene} | {s.qtype[:4]} | {q} | {s.parse_notes} |")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "date": date.today().isoformat(),
        # gt_battery has no non-default calibration path (score_scene never threads a
        # Thresholds/Calibration — every scorer uses DEFAULT_THRESHOLDS, i.e. the default
        # calibration's geometry — and there is no --calibration flag), so provenance
        # stamps the default calibration via collect_provenance's own fallback. This is
        # the single documented calibration path for this tool.
        "provenance": collect_provenance("gt_battery", argv),
        "answer_key_status": answer_key_status,
        "n_questions": len(scores),
        "scenes": scenes,
        "missing_scenes": missing,
        "aggregate": agg,
        "scores": [asdict(s) for s in scores],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return md_path, json_path


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def _num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="core.runner.gt_battery",
        description="Score the training questions against ground-truth VLA-3D scenes (REAL accuracy).",
    )
    ap.add_argument(
        "--groundtruth", required=True,
        help="Unity root dir containing per-scene folders (matched to questions.json scenes).",
    )
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="questions.json path")
    ap.add_argument(
        "--questions-dir", default=str(DEFAULT_QUESTIONS_ROOT),
        help="dir holding <scene>/trajectory_q*.ply (default: the challenge questions dir)",
    )
    ap.add_argument(
        "--answers", default=str(DEFAULT_ANSWERS),
        help="true numerical answer key (arch-F3); missing file -> true fields left null",
    )
    ap.add_argument("--out", default=None, help="output dir (default reports/gt_battery_<date>/)")
    ap.add_argument("--scenes", default=None, help="comma-separated scene subset")
    ap.add_argument(
        "--no-drive-if", action="store_true",
        help="skip driving instruction-following paths (score IF as unscored)",
    )
    ap.add_argument(
        "--no-spawn-hint", action="store_true",
        help="IF realism knob: spawn at the scene centroid instead of the GT-matched "
             "start, so exploration cost from a neutral pose is visible (IF-F2).",
    )
    ap.add_argument(
        "--no-walls", action="store_true",
        help="IF-F2 wall-realism escape: disable interior walls derived from each "
             "scene's traversable_area.ply, reproducing the old boundary-only mirror "
             "costmap (object obstacles + outer boundary, no interior walls).",
    )
    ap.add_argument(
        "--unity-scenes-ros2-root", default=str(DEFAULT_UNITY_SCENES_ROS2_ROOT),
        help="root holding <scene>/<scene>/traversable_area.ply, used to derive "
             "interior walls (IF-F2 wall realism); ignored with --no-walls.",
    )
    ap.add_argument(
        "--derive-tol", action="store_true",
        help="issue #70: derive the leg-arrival tolerance from THIS run's own "
             "p95 fit residual across scenes (a cheap no-driving pre-pass), "
             "instead of the frozen nominal core.groundtruth.scoring.LEG_ARRIVAL_TOL_M.",
    )
    args = ap.parse_args(argv)

    scenes = [s.strip() for s in args.scenes.split(",")] if args.scenes else None
    out_dir = args.out or (DEFAULT_OUT_ROOT / f"gt_battery_{date.today().isoformat()}")

    scores, missing = run_gt_battery(
        args.groundtruth,
        questions_path=args.questions,
        questions_dir=args.questions_dir,
        answers_path=args.answers,
        scenes=scenes,
        drive_if=not args.no_drive_if,
        no_spawn_hint=args.no_spawn_hint,
        walls=not args.no_walls,
        unity_scenes_ros2_root=args.unity_scenes_ros2_root,
        derive_tol=args.derive_tol,
    )
    if not scores:
        print(f"gt_battery: no scenes found under {args.groundtruth} (missing={missing})")
        return 1
    # Classify the answer key so the report topline distinguishes a corrupt key from a
    # legitimately absent one (issue #16). The loud stderr warning already fired inside
    # run_gt_battery's _load_answers; this is the pure classification for the report.
    answer_key_status = _answer_key_status(args.answers)
    # Stamp the actual invocation argv (fall back to the process args for a bare CLI run).
    stamp_argv = list(argv) if argv is not None else sys.argv[1:]
    md_path, json_path = write_report(
        scores, missing, out_dir, argv=stamp_argv, answer_key_status=answer_key_status
    )

    agg = aggregate(scores)
    n, o, i = agg["numerical"], agg["object_reference"], agg["instruction_following"]
    if n["n_with_true_answer"]:
        num_true_str = f"{_pct(n['true_accuracy'])} ({n['n_with_true_answer']} keyed)"
    elif answer_key_status == "unreadable":
        # Distinguish a corrupt/unparseable key from a legitimately absent one so the
        # yardstick can't vanish unnoticed in the stdout summary too (issue #16).
        num_true_str = "n/a[UNREADABLE]"
    else:
        num_true_str = "n/a"
    # Object-reference headline is instance-match / scoreability (same metric naming as the
    # markdown report), with IoU kept as a secondary diagnostic (issue #17).
    obj_rows = [s for s in scores if s.qtype == QType.OBJECT_REFERENCE.value]
    or_scored = [s for s in obj_rows if s.gt_target_id is not None]
    or_instance_match = sum(
        1
        for s in or_scored
        if s.our_target_id is not None and s.our_target_id == s.gt_target_id
    )
    # Headline OR metric matches the report: instance-match / scoreability
    # (IoU is a diagnostic pending real perception, see write_report).
    or_scored, or_instance_match = _or_instance_match(scores)
    or_instance_str = (
        f"{or_instance_match}/{len(or_scored)} (scoreable {len(or_scored)}/{o['n']})"
        if o["n"]
        else "n/a"
    )
    print(
        f"gt_battery: {len(scores)} questions / {len({s.scene for s in scores})} scenes  "
        f"num_true={num_true_str} "
        f"or_instance_match={or_instance_match}/{len(or_scored)} "
        f"or_iou={_num(o['mean_iou'])} "
        f"if_rubric={_num(i['mean_rubric_score'])} "
        f"if_thread_viol={i['total_threading_violations']} "
        f"if_avoid_viol={i['total_avoid_violations']}"
    )
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
