# Issue #77c — residual bucket, fresh post-#79 map + Fix 1

Branch `if77c-residual`, worktree `.claude/worktrees/if77c-residual`.
Baseline: `reports/gt_battery_main_post79` (credit 0.6111, headline 0.5889,
threading violations 3) — verified to match the task brief's cited numbers
exactly (`aggregate.instruction_following`).

## Phase 1a — fresh per-leg miss table (post79 data, no rerun)

Reproduced `conversion_probe.md`'s bucket methodology directly against
`reports/gt_battery_main_post79/gt_battery_results.json`'s
`leg_outcomes`/`leg_probe` (script:
`reports/issue77c_notes_work/bucket_post79.py`, full table
`reports/issue77c_notes_work/legs_post79.json`):

| bucket | count |
|---|---|
| `reached_in_order` | 45 |
| `cascade_victim` | 0 |
| `arrival_blocked` | 18 |
| `structurally_unreachable` | 9 |

Recomputed `mean_ordered_leg_credit` (macro-average per question) = 0.6111,
exact match to the reported aggregate — confirms the bucket table is a
faithful reconstruction, no drift from the #79 merge.

## Phase 1b — #78 blast-radius rescan against post-79 main

Adapted `reports/issue78_notes_work/scan_all_goto2.py` verbatim (script:
`reports/issue77c_notes_work/scan_goto_post79.py`, dump
`goto_scan_post79.json`). The same **6 goto legs immediately following a
`corridor_between` leg** are still flagged (`dist_to_rubric` UNCHANGED,
byte-identical to #78's original table) — #79 explicitly investigated and
refuted extending goal SELECTION itself (its own "2 dry attempts, both
refuted" section), so this is expected, not a regression.

Classification (per leg, using `reports/issue77c_notes_work/probe_goto_pocket.py`
— a generalized version of #79's `probe_goto_pinch.py` that runs the FULL
pinch-relax schedule for each of the 6 legs, not just `hotel_room_2`):

| scene | q | leg | dist_to_rubric | classification | evidence |
|---|---|---|---|---|---|
| `hotel_room_2` | 0 | 1 | 1.114 | **disconnected-geometry (structural)** | reconfirms #79: pocket never grows to include the anchor at ANY relax round (524→1040→1456→1465→1465 cells), and the best achievable distance-to-rubric ACTUALLY WORSENS under relax (1.114→1.178) — the room is disconnected by real, uninflated geometry beyond the gate, not a reachability-BFS artifact. Not currently credit-blocked (`reached_in_order=True` via tolerance luck, per #79). |
| `arabic_room` | 1 | 2 | 1.774 | still-broken, **not fixable at this leg's mechanism** | corridor leg1 (this leg's own predecessor) itself never threads (`threaded=False`, excess 3.08 m — see Phase 1c) — the route never even nears the gate, so no goal-resolution fix at leg2 can matter; gated upstream. |
| `hotel_room_1` | 1 | 2 | 2.409 | **fixed (Fix 1, this session)** | flips to `reached_in_order=True` |
| `livingroom_1` | 1 | 2 | 1.430 | already non-blocking | `reached_in_order=True` today (miss sits inside `tol_used`); its own predecessor corridor leg1 never threads either (excess 0.78 m, Phase 1c) |
| `home_building_2` | 1 | 1 | 1.516 | **attempted, refuted (see Fix 1 below)** | pinch-relax genuinely narrows the distance (1.516→0.821 best-round), but adopting the narrower goal broke the corridor leg0's OWN already-earned threading (a real interaction found and guarded against — see Fix 1's "regression found and fixed" note) — net: guard correctly keeps the OLD (wider) goal here, leg1 itself stays unfixed, no regression |
| `studio` | 1 | 2 | 3.338 | **fixed (Fix 1, this session)** | flips to `reached_in_order=True` |

Net: of the 6 blast-radius legs, 1 is structural (confirmed disconnected,
not currently costing credit), 2 are fixed this session, 2 are gated by an
UPSTREAM corridor leg that itself never threads (no fix possible at the
goto-goal-resolution mechanism), 1 (`home_building_2` leg1) has a real,
verified-reachable-but-currently-guarded-off improvement that a future,
more invasive fix (giving that specific leg's own corridor extension a way
to accept the narrower goal without losing its own threading — i.e. a
two-sided joint optimization, not a one-sided widen) could still land; not
attempted further this session (see Deferred).

## Phase 1c — 3 remaining threading violations, traced

| scene | leg | threaded | reached_in_order | excess (m) | classification |
|---|---|---|---|---|---|
| `arabic_room` | 1 | False | False | 3.077 | **never-approached** |
| `home_building_1` | 1 | False | False | 5.099 | **never-approached** |
| `livingroom_1` | 1 | False | False | 0.778 | **never-approached** |

All 3 are `structurally_unreachable` (goal itself implausible against the
GT reference path, `dist_gt > tol`), not `arrival_blocked` — the driven
route never gets near the gate at all, as opposed to the ORIGINAL 3
dwell-class "attempted" corridors (`hotel_room_1` leg1, `hotel_room_2`
leg0, `studio` leg1 — all `reached_in_order=True` pre-#79) that #79 already
fixed. Traced each to its leg0 predecessor (same question, earlier leg):

- `arabic_room` leg0: `structurally_unreachable`, excess **4.14 m** — a
  gross goal-resolution miss (`dist_gt=6.34` vs `tol=2.51`) drags the whole
  route off course before the corridor leg is ever approached.
- `home_building_1` leg0: `structurally_unreachable`, excess **9.65 m** —
  same shape, even larger miss (`dist_gt=13.01` vs `tol=2.24`); this is the
  scene independently flagged in `conversion_probe_v2/probe.md`'s
  boundary-only-costmap bucket (d) (`home_building_1`'s wall-derivation
  fallback).
- `livingroom_1` leg0: only a hair over (excess 0.039 m) but the corridor
  leg1 itself still misses by 0.78 m — this is `conversion_probe_v2`'s
  other (d)-bucket scene (also wall-unavailable).

**Conclusion: all 3 remaining threading violations are conversion-gated
(never-approached), not a new attempted-and-failed threading mechanism.**
This matches `conversion_probe_v2/probe.md`'s original finding verbatim
(same 3 legs, same "not attempted at all (0.4–5.1 m away)" classification)
— #79's fix cleared every ATTEMPTED corridor case; the residual 3 need
their own upstream goal-resolution/costmap fix (already tracked separately
as `arabic_room`/`home_building_1`/`livingroom_1`'s pre-existing
"boundary-only-costmap" (d) mechanism from `conversion_probe_v2`, unowned,
out of `#77c`'s scope as a NEW threading fix — there is nothing
threading-specific left to fix).

## Phase 2 — fix ledger

### Fix 1 — goto-leg goal resolution widens through a just-threaded gate (nav/heads)

**Root cause (per #78, reconfirmed).** `InstructionHead._goto_point`
resolves a GOTO leg's floor target via `Costmap.reachable_mask`'s plain
(non-pinch) BFS from the grounding pose. A GOTO leg immediately following a
`CORRIDOR_BETWEEN` leg sees only the NEAR-SIDE pocket of that leg's
just-threaded gate — the far side, where the goal legitimately sits, is
invisible to the plain flood — exactly the asymmetry `plan_through`
resolves for ROUTING via `_pinch_costmap`/`last_gate` (#79) but goal
SELECTION never saw. #79 investigated and explicitly refuted a naive fix
for ONE leg (`hotel_room_2`) and stopped there, recommending #78 stay open.
This session generalized the SAME probe (full relax schedule, not #78's
single-disc check) to all 6 blast-radius legs and found the naive refutal
does NOT generalize — 2 of the other 5 legs (`hotel_room_1` leg2, `studio`
leg2) genuinely close under the full schedule; only `hotel_room_2`
specifically is disconnected-by-real-geometry.

**Fix.** `_goto_point` now accepts an optional `prev_gate` (threaded from
`_ground_legs`/`_ground_one`, which already track `prev_kind`; only passed
when the immediately preceding leg is `CORRIDOR_BETWEEN`). When the plain
BFS result isn't good enough, `_goto_point_pinch_relax` retries
`nearest_reachable_point` through that SAME gate over the SAME
`PINCH_DISC_M`/`MAX_PINCH_RELAX_ROUNDS`/`PINCH_RELAX_GROWTH`/
`PINCH_CORRIDOR_HALF_W_M` relax schedule `plan_through` itself uses (no new
free parameter — reuses `core.nav.planner`'s existing calibration seams and
`_pinch_costmap` helper verbatim), keeping whichever candidate (plain or
any relaxed round) lands closer to the anchor's own centroid.

**Regression found and fixed during integration (not shipped naively).**
The first cut of this fix (widen unconditionally, no validation) measured
credit 0.6222/headline 0.600 but **regressed `home_building_2`'s own
already-earned corridor threading** (`threaded True->False`,
`reached_in_order True->False` on its OWN leg0): moving the FOLLOWING goto
leg's resolved goal to a farther-but-truer point made
`plan_through`'s `_gate_extension_keeps_route_planned` lookahead guard
(shared between the corridor leg's own gate-crossing extension and the
goto pinch fallback, #79) correctly refuse to commit the corridor's own
extension, since the new goal is no longer reachable from the extension
candidate the way the old (wronger but "closer" in graph-distance) goal
was. Traced to source with a temporary debug print in
`_gate_crossing_extension` (removed before commit; net diff to
`planner.py` is zero) confirming `_gate_extension_keeps_route_planned`
flips `True->False` between the old and new `home_building_2` leg1 goals,
at the exact call site `plan_through` will use.

**Fix, corrected.** `_goto_point_pinch_relax` now validates each candidate
BEFORE adopting it: calls `core.nav.planner._gate_crossing_extension`
directly (the SAME function `plan_through` will call, imported not
duplicated) with the gate's own `usable_gate_point` as a stand-in
extension origin and the candidate as the next leg's target; a candidate
that would make the corridor leg's own extension fail is rejected, falling
back to the previous (narrower but non-regressing) result. This reuses
existing, already-shared planner helpers (`_gate_crossing_extension`,
`_gate_extension_keeps_route_planned`, `usable_gate_point`) rather than
reimplementing their logic — no new duplicated pinch semantics, addressing
the exact risk #78/#79's notes flagged about drift between two modules.

Also hardened against a pre-existing, previously-unexercised hazard: `cm`
(the head's `_costmap`) can be transiently stale relative to `self.grid`
mid-tick (built before this tick's terrain ingestion grew the live grid;
`_build_route` reconciles it later in the same tick) — `reachable_mask`'s
BFS never touched this because it only walks cells it can reach, but
`_pinch_costmap` vectorizes over the full grid shape unconditionally. Added
a cheap `cm.grid.shape != cm.capsule_blocked.shape` guard that skips the
relax (falls back to the plain result) rather than crashing — a real bug
this session's own new code path was the first to expose (never touched
`_pinch_costmap` from grounding time before), not a `src/` regression.

**Files.** `src/core/heads/instruction.py` (`_ground_legs`/`_ground_one`
thread `prev_kind`/`prev_gate`; `_goto_point` + new
`_goto_point_pinch_relax`). `src/core/nav/planner.py`: untouched net (debug
print added and removed during investigation; `git diff` is empty).

**Measurement.** Fast tier: green. Full gate
(`pytest -m "" -q` from `src/`): green (`reports/issue77c_notes_work` — see
"Tests" below). Integrated battery
(`reports/gt_battery_if77c_fix1_gotopinch`):

| metric | post79 (baseline) | fix1 (this) | Δ |
|---|---|---|---|
| `mean_ordered_leg_credit` | 0.6111 | **0.6333** | **+0.0222** |
| `mean_rubric_score` (headline) | 0.5889 | **0.6111** | **+0.0222** |
| `total_threading_violations` | 3 | 3 | 0 |
| `total_avoid_violations` | 0 | 0 | 0 |

Diffed EVERY question's `leg_outcomes`/`rubric_score` between baseline and
this fix: **2 questions changed, both improved, zero regressions** across
all 75 questions:

| scene | question (truncated) | leg | rubric before→after |
|---|---|---|---|
| `hotel_room_1` | "First, go near the bedside table…take the path between the TV and the bed…" | leg2 (goto) | 0.667 → 1.000 |
| `studio` | "First, go to the vase…take the path between the couch and the table…" | leg2 (goto) | 0.667 → 1.000 |

`home_building_2`'s leg1 (the 3rd theoretically-fixable blast-radius leg)
stays unchanged — the validation guard correctly keeps its OLD goal (see
above), so it neither improves nor regresses.

### Commit

`heads: widen goto-leg goal resolution through a threaded corridor gate (#77)`

## Structural ledger (final count)

Legs adjudicated OUT as structural (disconnected mirror-world geometry /
mesh-coverage gaps / frame-fit defects) — not chased further, honest
ceiling documented:

| scene | legs | mechanism | evidence |
|---|---|---|---|
| `home_building_1` | 2 (coffee-table-kettle question, both legs) | mesh-coverage gap (~3 m beyond `traversable_area.ply` extent) | #76's Phase 1b measurement, reconfirmed unaffected by wall-derivation (#77's notes) |
| `livingroom_3` | 3 | frame-fit-unfittable (`fit_residual_m` 1.81 > 1.0 gate, meth-F11) | pre-existing, `conversion_probe_v2` (e)-frame bucket |
| `hotel_room_2` | 1 (leg1, "lamp closest to the fireplace") | disconnected-by-real-geometry: room fragments at the bench/bed gate even under the full pinch-relax schedule; `nearest_reachable_point`'s best achievable distance to the true goal WORSENS under relax | #79 (2 refuted attempts) + this session's reconfirmation via the generalized probe (`reports/issue77c_notes_work/probe_goto_pocket.py`) |

**Structural ledger total: 6 legs** (2 + 3 + 1) across 3 scenes — this is
the honest, currently-known "never fixable without a materially bigger
mechanism" floor for the residual bucket. (`home_building_1`'s and
`arabic_room`'s upstream leg0 goal-resolution misses feeding the 2
never-approached threading violations in Phase 1c are NOT added here —
those are large but not proven geometrically impossible, just unowned;
see Deferred.)

## Tests

Fast tier (`pytest` from `src/`, no `-m`): green throughout (checked after
each edit, including the reverted debug print in `planner.py`).

Full gate (`timeout 590 <venv>/bin/python -m pytest -m "" -q` from `src/`):
**green, exit code 0** — run for the shared-path touch (`core/nav/planner.py`
was read and instrumented for debugging, though its net diff is zero; the
shipped fix touches `core/heads/instruction.py`, which drives leg-goal
resolution feeding the shared `plan_through` path).

## Integrated numbers (final, this session)

Credit **0.6333**, headline **0.6111**, threading violations **3**, avoid
violations **0** — up from post79's 0.6111 / 0.5889 / 3 / 0. Short of the
0.65 stop target; continuing/stopping decision below.

## Deferred / next steps

1. **`home_building_2` leg1's own goal fix** — the widened, more-accurate
   goal for this leg (dist-to-rubric 1.516→0.821) is real and reachable,
   but adopting it costs the SAME question's leg0 its own threading credit
   (a genuine trade-off, not a bug in either mechanism). A real fix needs a
   JOINT resolution — e.g. re-deriving the corridor leg's own extension
   candidate AFTER the following goto leg's goal is known, rather than the
   current one-pass, order-dependent grounding — a materially bigger
   change than this session's single-function widen, and touches the
   shared `plan_through` tail every leg kind uses. Not attempted.
2. **The 3 never-approached threading violations' upstream cause**
   (`arabic_room`/`home_building_1` leg0 gross goal-resolution misses,
   `livingroom_1`'s wall-unavailable costmap) is `conversion_probe_v2`'s
   pre-existing, unowned "(d) boundary-only-costmap" mechanism — not a
   threading-specific defect, out of this issue's threading-classification
   scope. Worth its own lane if revisited.
3. Everything else from #77/#77b/#78/#79's own "Deferred" sections not
   superseded by this session's Phase 1 remains open (the broad "(e) other"
   residual, `_goto_point`'s separate coincident-point bug for
   `hotel_room_2` leg1's OWN resolution — distinct from the pinch-relax
   mechanism fixed here).
