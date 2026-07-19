# Issue #77 — decisive residual bucket: adjudication + fix ledger

Branch `if77-residual`, worktree `.claude/worktrees/if77-residual`.
Baseline: `reports/gt_battery_main_post75` (credit 0.5444, headline 0.4833,
threading violations 7).

## Phase 1 — rebuilt per-leg miss table (post75 data, no rerun)

Reproduced `reports/conversion_probe.md`'s bucket methodology directly
against `reports/gt_battery_main_post75/gt_battery_results.json`'s
`leg_outcomes`/`leg_probe` (script: `reports/issue77_notes_work/legs.json`
holds the full 72-leg table; generation script was ephemeral, logic
reproduced below):

- `reached_in_order`: 40/72 (macro-average over 30 questions gives
  `mean_ordered_leg_credit=0.5444`, matches the battery aggregate exactly).
- `arrival_blocked`: 22, `structurally_unreachable`: 10 — 32 non-in-order
  legs total.

### Adjudication — home_building_1's 2 mesh-gap legs, OUT (structural)

Per #76's Phase 1b measurement (`reports/issue76_notes.md`): `home_building_1`'s
"coffee table with kettle" question (2-leg goto), object-frame goal x ≈ 7.55,
this scene's `traversable_area.ply` only covers x ∈ [-16.04, 4.51] through the
fitted frame — a genuine ~3 m mesh-coverage gap, unaffected by wall-derivation
(fails identically with interior walls forced on or off, and independent of
`_gt_footprint_bounds`'s much larger outer rectangle, which applies in both
cases). In the current post75 data this is:

| scene | qtext | leg | bucket | dist_gt(m) | excess(m) |
|---|---|---|---|---|---|
| home_building_1 | "coffee table with the kettle..." | 0 | structurally_unreachable | 14.824 | 1.132 |
| home_building_1 | "coffee table with the kettle..." | 1 | structurally_unreachable | 13.028 | 3.279 |

Both `dist_gt` values are far beyond any plausible tolerance band (13-15 m
vs. a ~1.7-3.1 m tolerance) — consistent with #76's finding that the mirror
world cannot host this goal at all. **Adjudicated OUT as structural for the
offline yardstick — not chased.** (`home_building_1`'s OTHER question,
"nightstand with a clock... corridor", is a DIFFERENT 3-leg question and
stays in the residual pool below — #76's ledger only cleared the coffee-table
legs as mesh-gap; the nightstand/corridor legs have no such confirmed
structural defect.)

`livingroom_3`'s 3 legs (frame-fit-unfittable, meth-F11, `fit_residual_m`
1.81 > 1.0 gate) are the OTHER pre-existing, separately-tracked structural
defect noted by probe-v2 — also left alone (out of #77's scope, not a new
finding here).

### Ranked remaining misses (30 legs, excess ascending)

Full table: `reports/issue77_notes_work/legs.json`. Top of the ranking
(cheapest first): `studio` leg1 (0.022), `arabic_room` leg2 (0.028),
`livingroom_1` leg0 (0.039), `livingroom_4` leg0×2 questions (0.100, 0.215),
`office_2` leg1×2 questions (0.127, 0.184), `hotel_room_1` leg0 (0.170),
`studio` leg0 (0.314) — an 8-leg **near-miss cluster, excess < 0.32 m**,
matching the brief's prediction almost exactly (shared mechanism: dwell
release, see Fix 1 below). Everything past that clusters into the same
broad, no-signature "(e) other" bucket probe-v2 already characterized (no
new sub-signature found).

### Threading — 3 dwell-class attempted-corridor legs

`hotel_room_1` leg1, `hotel_room_2` leg0, `studio` leg1 — all
`reached_in_order=True` (rubric-credited arrival at the gate midpoint) but
`threaded=False` (driven trajectory never literally crosses the gate p0-p1
segment). Traced to `core/nav/breadcrumbs.py`'s dwell-release mechanism
(root cause below, Fix 1 attempted this exact mechanism).

## Phase 2 — fix ledger

### Fix 1 — leg-goal dwell release used the wrong tolerance concept (nav)

**Root cause.** `BreadcrumbFollower._leg_ceiling()`/`_advance_leg_goals()`
(issue #74) caps crumb selection at a pending leg goal until the pose
dwells within `LEG_GOAL_ARRIVAL_TOL_M` of it. That constant was set to
`core.groundtruth.arrival.NOMINAL_ARRIVAL_TOL_M` (~1.75 m) — the rubric's
own frame-fit-derived CREDIT tolerance, reused on the stated theory that the
dwell gate should never be a stricter bar than the rubric already applies.

That theory is correct for whether a leg gets CREDIT, but the dwell gate
does a second, unrelated job: it decides when the follower is ALLOWED TO
STOP AIMING at the leg goal and let crumb selection move on. Releasing that
gate from up to ~1.75 m away — measured against `path[ceiling]`, not
against the scoring goal directly, and for `corridor_between` legs almost
double the observed miss distances (0.55-1.44 m) — let the crumb scan jump
to a farther, still-LOS-clear point beyond the goal before the vehicle had
physically closed on it. For non-terminal/STOP legs this capped the
vehicle's actual closest approach to roughly the dwell radius, producing
exactly the < 0.3 m near-miss cluster surfaced in Phase 1 (dwell already
"good enough" to release, so the follower quit closing the last fraction of
a metre that would have crossed into the rubric's own tolerance). For
corridor legs it meant crumb selection could abandon the beeline to the
gate midpoint before the vehicle had traversed it, explaining "arrives near
the gate, never crosses" independent of any threading-logic bug (confirmed
in `reports/conversion_probe.md`'s prior investigation: the #64 nudge
guard is correctly not engaging — this was never a nudge-guard issue).

**Fix.** `LEG_GOAL_ARRIVAL_TOL_M` now equals `REACH_M` (0.8 m) — this SAME
follower's own already-established "close enough to a path point to
consider it reached" radius, already governing the ordinary within/overshoot
progress-index advance a few lines below in the same class. This is not a
new tuned threshold: it reuses an existing coded constant from the same
object for the same underlying question ("has the follower physically
arrived at this path point"), rather than importing a second, unrelated
tolerance concept (GT frame-fit residual) that has nothing to do with the
follower's own steering precision. Per the generalization protocol
(`docs/calibration.md`): no new free parameter introduced, and the change
can only make the follower drive CLOSER before releasing a leg goal, never
farther — CREDIT semantics are untouched (still keyed to the rubric's own
`tol`/`tol_used`, computed entirely in `core/groundtruth/scoring.py`, not
touched here).

File: `src/core/nav/breadcrumbs.py` (`LEG_GOAL_ARRIVAL_TOL_M` constant +
docstring; no other lines changed). Frozen surfaces
(`core/groundtruth/scoring.py`/`arrival.py`, `core/geometry/toolbox.py`
resolve semantics, `core/llm`) untouched.

**Measurement.** Full fast tier: green (`pytest` from `src/`, no failures).
Integrated battery
(`gt_battery --groundtruth ../data/vla3d/Unity --out
../reports/gt_battery_if77_fix1_dwelltol`):

| metric | post75 (baseline) | fix1 (this) | Δ |
|---|---|---|---|
| `mean_ordered_leg_credit` | 0.5444 | **0.6000** | **+0.0556** |
| `mean_rubric_score` (headline) | 0.4833 | **0.5389** | **+0.0556** |
| `total_threading_violations` | 7 | 7 | 0 |
| `total_avoid_violations` | 0 | 0 | 0 |

4 legs flipped `reached_in_order: False -> True` (all from the predicted
near-miss cluster, all non-corridor):

| scene | question | leg | dist_driven before -> after (m) |
|---|---|---|---|
| hotel_room_1 | "go near the bedside table closest..." | 0 | 2.506 -> 2.261 |
| livingroom_4 | "go near the chair closest to the bookcase..." | 0 | 2.228 -> 1.341 |
| office_2 | "first, go to the trash can near the cabinet..." | 1 | 2.006 -> 1.134 |
| studio | "go to the vases on the cabinet below the..." | 0 | 2.173 -> 1.760 |

No leg regressed (checked full before/after diff over all 72 legs — every
changed row is a strict improvement, zero flips the other direction).

**Threading — not fixed by this change.** All 3 attempted-corridor legs
still show `threaded=False` after Fix 1, though their approach distance
shrank (`hotel_room_1` leg1: 0.579 -> 0.55 m; `hotel_room_2` leg0: 1.147 ->
0.754 m; `studio` leg1: 1.439 -> 0.631 m — closer, but the driven trajectory
still never crosses the gate p0-p1 line). Releasing the dwell gate sooner
was necessary but not sufficient: the crumb-selection scan, once released,
still has no notion that "the next crumb should be on the FAR side of the
gate" — it just resumes ordinary farthest-within-lookahead selection, which
can pick a point that curves around the gate rather than one that forces a
literal crossing. A real threading fix needs the follower (or route
assembly) to treat a corridor leg's ceiling release as "crossed", not just
"dwelled near", which needs gate geometry threaded into
`BreadcrumbFollower` (currently only `leg_goal_indices`, no gate segment) —
a larger, more invasive change than Fix 1's single-constant swap. **Not
attempted this session** given the 2-hour budget and Fix 1 already landing
the credit/headline win from this same investigation; flagging as the
concrete next step for threading specifically (see "Deferred" below).

### Commit

`nav: derive leg-goal dwell release from the follower's own reach radius (#77)`
— see git log in this worktree.

### Fix 2 (attempted, DRY — reverted) — literal gate-crossing release for corridor legs

**Hypothesis.** Fix 1 closed the near-miss cluster but left all 3 attempted-
corridor legs `threaded=False`. Theory: the dwell-proximity release (even at
the now-tighter `REACH_M`) is still a PROXIMITY test, not a LITERAL crossing
test — `core.geometry.toolbox.threading_check` requires an actual
segment-intersection between a driven-trajectory edge and the gate's own
`(p0, p1)` line (the same unnudged anchor-to-anchor axis
`core.heads.instruction._ground_one` already computes via `TB.corridor_gate`
and passes as a `corridor_between` leg's `geom`), so a follower that releases
on proximity to the (possibly obstacle-nudged) via-point can move on before
ever literally crossing that line.

**Implementation (reverted).** Threaded `plan_through`'s already-known gate
segments (`geom` for each `corridor_between` leg — literally the same
`(p0, p1)` scoring's `threading_check` uses) through a new opt-in
`record_corridor_gates=True` return value, wired to a new
`BreadcrumbFollower.leg_goal_gates` field; a corridor leg's ceiling then
released only once an incremental driven-motion segment
(`core.geometry.primitives.segments_intersect_2d`) was detected crossing its
own gate, instead of dwell-proximity. Files touched: `core/nav/planner.py`
(opt-in third return value, `_unreachable` helper), `core/heads/instruction.py`
(pass `record_corridor_gates=True`, wire `leg_goal_gates`),
`core/nav/breadcrumbs.py` (`leg_goal_gates`, `_prev_pose`,
`_crossed_gate_ptrs`, `_update_gate_crossings`, gated `_advance_leg_goals`).
Fast tier: green.

**Measurement — dry.** Integrated battery
(`reports/gt_battery_if77_fix2_gatecross`, since deleted with the revert):
credit 0.6000 (unchanged), headline 0.5389 (unchanged), threading violations
7 (unchanged). All 3 corridor legs' `dist_driven` moved marginally closer
(e.g. `hotel_room_1` leg1 0.55 -> 0.419 m) but **none crossed** —
`threaded` stayed `False` for all 3, identical to Fix 1. Zero leg outcomes
changed anywhere in the battery.

**Why it didn't land, and why reverted rather than kept as a no-op.** The
underlying `astar` path DOES cross the gate line by construction
(`plan_through` already rejects any corridor-leg plan that fails
`path_crosses_gate` against this exact segment, falling back to the pinch
overlay). But `BreadcrumbFollower` doesn't drive the literal path polyline —
crumb selection picks the farthest LOS-clear point within lookahead, which
capped at the ceiling still just aims the vehicle AT the (single) via-point,
never past it, and whatever downstream kinematics/controller loop drives the
pose toward that crumb apparently converges asymptotically close (0.4-0.75 m
observed) without the incremental per-tick motion segment ever numerically
intersecting a zero-width line — a controller-precision limit outside
`BreadcrumbFollower`'s own view (it only sees the poses it's handed; nothing
in this class controls how close the actual stepping gets). Requiring a
literal crossing at the follower level is therefore necessary but provably
not sufficient on its own — with no measurable improvement and real added
surface/complexity (2 new fields, a new cross-module return-value contract,
a new import), this fails the "keep only improvements" bar and was
**reverted** (`git checkout` back to the Fix 1 commit; verified clean diff
and fast tier still green after revert). Counts as 1 dry fix (not yet the
2-consecutive-dry-fix stop condition, but combined with the session's time
budget this is where Phase 2 stops).

## Stop condition

Ending Phase 2 here for this session: Fix 1 landed the largest single
identified mechanism win in the residual bucket (+0.0556 credit/headline,
4 legs, matching the near-miss cluster prediction exactly) with a
one-constant, fully justified change. Fix 2 (threading) was a legitimate,
principled attempt at the SAME investigation the brief asked for
("trace WHY dwell doesn't produce a crossing") but measured dry and was
reverted — 1 dry fix, not yet the 2-consecutive stop condition, but combined
with the ~2-hour session budget this is where Phase 2 stops. Headline after
Fix 1 (final, shipped state): **0.5389** — short of the 0.60 stop target, so
this is a time-budget stop, not a dry-fix stop.

## Deferred / next steps

1. **Threading fix (3 legs, ~0.04 headline) — needs controller-level work,
   not follower-level.** Fix 2's finding narrows this: the follower-level
   release mechanism is not where the gap lives — the driven trajectory's
   own poses never get close enough to numerically cross a zero-width gate
   line even when the follower keeps aiming at the exact via-point
   indefinitely. The next attempt should look at whatever produces the pose
   stream between crumb commands (the FSM/controller stepping this
   follower drives, outside `core/nav/breadcrumbs.py`'s own surface) for a
   convergence/precision limit, or consider widening the CREDITED
   "threaded" check itself (out of reach — `core/geometry/toolbox.py`'s
   `threading_check` sits inside the frozen "resolve semantics" boundary
   only in the loose sense that it's the same file; it is NOT part of
   `resolve()`'s own semantics and may be back in scope pending
   clarification, but was not touched this session out of caution).
2. **Remaining near-miss / broad "(e) other" legs** (22 of the ranked 30,
   excess 0.66-9.65 m after removing the 8-leg near-miss cluster Fix 1
   landed) — no new signature found beyond probe-v2's "genuine
   navigation/planning shortfall, no single mechanism" ruling. Re-rank
   against the fix1-integrated battery
   (`reports/gt_battery_if77_fix1_dwelltol`) before further chasing, since
   Fix 1 already shifted several `dist_driven` values.
