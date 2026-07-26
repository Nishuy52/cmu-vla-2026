# Issue #77 — instruction-following drive-precision residual

Session scope: drive/controller/waypoint-conversion precision surfaces
(`core/nav`, `core/heads` non-explore, probe/replay tooling). Excluded:
`core/nav/frontiers.py`, `core/nav/occupancy.py`, `core/nav/exploration.py`,
`core/heads/explore_step.py`, `core/perception`, `ros_adapter`.

## Starting state (before this session)

`git log` shows issue #77 already went through fold-shortcut (#72),
leg-boundary-skip (#74), salience mispick (#75), dwell-release retuning,
goal-widening, a joint-leg design doc (deferred), a 9-leg structural ledger
adjudication (`reports/issue77d_notes.md`), a terminal 14-leg-pool finding
that traced the entire remaining fixable pool to ONE mechanism — grounding-
pose BFS disconnection from dense furniture-AABB stamping
(`reports/issue77e_notes.md`) — and, most recently, the gated
`pre_grounding_movement_plan` Stage 0/1/3: mirror-truth audit, OBB
rasterization, border-padding fix (Stage 1 piece 1), and GT-trajectory
corridor carving (Stage 1 piece 2, commit `8324db5`).

**That carve was a big win**, measured and committed in `b8dbb94`
(`reports/gt_battery_main_post_carve/`): headline rubric **0.611 -> 0.744**,
zero regressions, over the full 75-question / 15-scene battery.

This session picked the thread back up from that post-carve state (0.744)
to find further mechanism fixes toward 0.8, per the issue's per-leg
top-N-by-excess-distance / fix-one-at-a-time / stop-on-two-dry-fixes
protocol.

Environment note: this worktree has no `data/`/`upstream/` (both
`.gitignore`d, and git worktrees don't share ignored files with the main
checkout). Symlinked them read-only from the main checkout
(`/home/jason/cmu_ws/cmu-vla-2026/{data,upstream}`) for the duration of the
tracing session, removed them again before finishing — no writes to the
main checkout, no repo state changed by this.

## Per-leg mechanism map (post-carve residual, 12 non-perfect `inst` rows)

| Scene | Q (short) | rubric | legs | Mechanism | Status |
|---|---|---|---|---|---|
| arabic_room | "go near the stool..." | 0.50 | 1/2 | **Goal-clamp / frame-fit mismatch** (see below) | traced this session — not fixable on my surface |
| arabic_room | "...potted plant... path between the two columns..." | 0.00 | 1/3, 1 thread-viol | Same goal-clamp mismatch, poisons the whole route (corridor leg downstream never reached either) | traced this session — one of the issue's "3 corridor threading" cases |
| chinese_room | "...tea table..." | 0.50 | 1/2 | Same goal-clamp mismatch | traced this session |
| home_building_1 | "coffee table with kettle..." | 0.00 | 0/2 | Wall-unavailable scene (frame residual 0.88 m > 0.8 m gate) | **already adjudicated STRUCTURAL** (issue77d: "wall-unavailable-scene shape deviation") |
| home_building_1 | "nightstand... corridor..." | 0.00 | 1/3, 1 thread-viol | Same wall-unavailable scene | **already adjudicated STRUCTURAL** |
| home_building_2 | "path between sofa/coffee table..." | 0.67 | 3/3, 1 thread-viol | Gate real-geometry mostly blocked; path goes around, not between | **already adjudicated & DEFERRED** (issue77d: "85% of gate physically blocked; small payoff, shared-path risk"; design doc in `reports/issue77d_notes.md`, not implemented) |
| livingroom_1 | "lamp... corridor..." | 0.00 | 1/3, 1 thread-viol | Wall-unavailable scene | **already adjudicated STRUCTURAL** |
| livingroom_2 | "chair near window..." | 0.50 | 1/2 | Same goal-clamp mismatch | traced this session |
| livingroom_3 | "path near TV... pillow..." | 0.50 | 1/2 | Frame unfittable (rigid two-point residual floor 1.04 m > 1.0 m gate) | **already adjudicated STRUCTURAL** (meth-F11, `issue77*_notes.md`) |
| livingroom_3 | "stool... cabinet..." | 0.67 | 2/3 | Same frame-unfittable scene | **already adjudicated STRUCTURAL** |
| loft | "cup near TV remote, avoid path..." | 0.00 | 0/1 | Same goal-clamp mismatch | traced this session |

6 of 12 rows were already carried as adjudicated structural/deferred from
prior sessions (unchanged by this one). The other 6 (arabic_room x2,
chinese_room, livingroom_2, loft, and — mechanistically distinct —
home_building_2) were traced fresh this session.

## New tracing this session: the goal-clamp / frame-fit mismatch

Tool: `reports/issue77_notes_work/trace_v2.py` (adapted from the existing
`reports/issue77e_notes_work/trace_leg.py` for the post-carve costmap, which
now needs `carved_cells` threaded through `_synthetic_from_gt`).

For every one of arabic_room (both questions), chinese_room, livingroom_2,
and loft, the pattern is identical and clean:

- `min_dist_driven_to_goal(head_goal) == 0.0000` — **the driven trajectory
  lands EXACTLY on the head's own resolved goal, every time.** The
  follower/controller/waypoint-conversion chain (my owned surface) has zero
  execution error on any of these legs.
- `head_goal` (what the follower actually drives to) sits **2.5 m to 8.8 m**
  away from the correctly-resolved anchor's own centroid (e.g. arabic_room
  q0: anchor = stool #34 at `(-3.74, -1.67)`, but `head_goal` clamped to
  `(-6.25, -1.65)` — the anchor label/instance is right, the point driven to
  is not).
- `nearest_reachable_point(head_goal) == head_goal` (distance 0) confirms
  the clamp is self-consistent with the costmap's own BFS-reachable mask —
  this is exactly the anchor's cell being **BFS-unreachable from spawn**
  through the mirror's derived/stamped geometry, so `_goto_point` snaps the
  goal to the nearest cell the BFS *can* reach, which can be on the far
  side of the room.
- For arabic_room specifically, I went one level deeper: the scene's own
  GT reference trajectory (the human-driven path IF-F2 diagnostics are
  compared against) never comes within 6.15 m of the resolved stool's
  centroid at all — its whole bounding box sits on the opposite side of the
  room (`x in [2.2, 7.0]` vs stool `x = -3.74`). The *rubric*'s own leg goal
  (`(-3.14, -1.98)`, derived from the same object positions/frame fit) is
  also nowhere near where the human actually walked. That is consistent
  with `frame_residual = 0.574` — a large, non-trivial per-scene fit
  residual — feeding position error into both the rubric goal and our own
  resolution, on top of the BFS-disconnection.

This is the exact mechanism issue #77e already named and terminal-findinged
("grounding-pose BFS disconnection from dense furniture-AABB stamping...
correct instances resolved, anchors unreachable from the single spawn-time
BFS") and which the gated `pre_grounding_movement_plan` exists to address
architecturally. The Stage 1 carve fixed this for anchors that happen to
sit ON the GT reference trajectory (hence the big 0.611->0.744 jump); these
6 legs' anchors do not (confirmed directly for arabic_room above), so the
carve correctly did not touch them. `reachable_mask`/`nearest_reachable_point`
live in `core/nav/costmap.py` (my surface, not excluded) but the actual fix
— making BFS reachability match physical reality without re-deriving the
whole mirror from a movement-based pass — is exactly what
`docs/proposals/pre_grounding_movement_plan.md` Stage 2 (not yet built) is
for, not a local waypoint-conversion tweak.

## One mechanism fix attempted and measured this session (NOT committed — regresses)

Stage 3's landing commit (`8402939`/`d9c99f0`/`9abe348`) explicitly flagged
"Goes live in the battery via Stage 4 minimal parity hooks (next)" — i.e.
the goal-credibility withhold gate (`InstructionHead._clamp_gated_prefix_len`,
`core/heads/instruction.py:695`) is a no-op in `gt_battery` today because
`_run_instruction_head`/`_drive_if_trajectory` never construct
`InstructionHead` with a `budget_frac`/`forced_assembly` hook, and
`_commit_forced()` defaults `True` with no hook (byte-preservation was the
explicit Stage 3 goal). I tried actually wiring those hooks for the battery
(`budget_frac=lambda: 0.0, forced_assembly=lambda: False` — Gate 3 always
armed, never overridden) against arabic_room q0 as a single-leg experiment
(not committed; see `/tmp/gate3_probe.py`, reconstructable from this note).

**Result: strictly worse.** With Gate 3 active, `_committable_prefix_len()`
returns 0 for the whole question — leg0's bad clamp (2.51 m > the 0.8 m
credibility bar) blocks the ENTIRE route from ever committing, so the
follower never gets built and the vehicle never moves at all
(`follower=None, path_len=0`). Today (Gate 3 inert), the route still
commits and drives past the bad leg0, picking up ordered credit on leg1 —
arabic_room q0 currently scores 0.50 (1/2). Arming Gate 3 without a real
re-grounding opportunity behind it (no exploration loop in the fully-
observed battery — that's `core/heads/explore_step.py`, explicitly out of
my ownership) has nothing to improve the clamp with; it can only ever
withhold, never recover, so it can only cost credit, not gain it. This
matches why Stage 3 was deliberately built byte-identical by default rather
than wired live in the battery.

This counts as one measured, negative mechanism-fix attempt. Given every
other traced leg in the residual pool independently confirms the same root
cause (BFS-disconnected clamp, already architecture-gated) or is already
adjudicated/deferred from a prior session, there is no second untried
mechanism candidate on my owned surface to spend a second attempt on — so
rather than force a token second fix, I'm stopping here and reporting the
finding, per "stop and say so with numbers rather than grinding."

## Score

Unchanged this session: **headline rubric 0.744** (0.611 pre-carve ->
0.744 post-carve, per `reports/gt_battery_main_post_carve/gt_battery_report.md`,
last committed by the prior session in `b8dbb94`). No fix from this session
was committed (the one attempted fix regresses and was reverted/never
applied to source).

## Fast tier

Green apart from pre-existing, unrelated environment gaps present before
this session touched anything:
- `tests/replay/*` (12 files) error at collection with
  `ModuleNotFoundError: No module named 'rosbags'` — a missing optional
  dependency in this environment, not a code defect.
- `tests/parsing/test_regex_full_set.py` (3 tests) error with
  `missing training questions at .../upstream/CMU-VLN-Challenge-2026/questions/questions.json`
  — the `upstream/` fixture tree is `.gitignore`d and not present in this
  worktree by default (see the Environment note above).
- `tests/runner/test_gt_battery.py::test_synthetic_scene_extra_wall_cells_mark_terrain_obstacle`
  fails on `assert near[0, 3] == WALL_HEIGHT` — a `float32` (terrain array,
  `core/mocks/synthetic_scene.py:343`) vs Python `float` (`WALL_HEIGHT =
  2.4`, `core/mocks/synthetic_scene.py:23`) precision mismatch
  (`np.float32(2.4) != 2.4`), present before any commit made in this
  session (confirmed: `synthetic_scene.py` untouched by every #77 commit in
  this branch's history) and unrelated to drive precision. Flagged, not
  fixed (outside this session's touched surface; a one-line
  `np.float32(WALL_HEIGHT)` cast in the test would resolve it but that's a
  test-authoring call for whoever owns that file, not a #77 mechanism).

All other collected tests pass (confirmed via
`pytest -q --deselect tests/runner/test_gt_battery.py::test_synthetic_scene_extra_wall_cells_mark_terrain_obstacle --ignore=tests/replay --ignore=tests/parsing/test_regex_full_set.py`).

## Stop-condition status

**Needs-full-battery-command: N/A (no fix to measure against the full
battery — nothing committed this session).** Score-relevant conclusion:
the post-carve 0.744 stands; 0.8 is not reachable from further grinding on
my owned surface (drive/controller/waypoint-conversion) without the
gated `pre_grounding_movement_plan` Stage 2 (BFS-reachability-vs-physical-
reality reconciliation) or a frame-fit-quality improvement (outside my
ownership — perception/frame-fitting). This corroborates, rather than
contradicts, the standing 77e verdict.

## Filed / to file

No new GitHub issue filed — every mechanism found in this session (goal-
clamp/BFS-disconnection, frame-fit residual, home_building_2's blocked
gate) is already tracked in the existing #77 comment thread and in
`reports/issue77d_notes.md` / `reports/issue77e_notes.md`. Posting a
comment on #77 summarizing this session's confirmation + the negative
Gate-3-hook-wiring result (so the next session doesn't re-attempt it) --
see the issue comment posted alongside this note.
