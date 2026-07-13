# Red-team review — Instruction-Following pipeline

**Date:** 12 Jul 2026
**Facet:** IF leg grounding, corridor/avoid geometry, plan_through → breadcrumbs, terminal answer, and the GT scorer vs the real rubric.
**Stakes:** IF = 70.6% of all points (180/255 training-equivalent; at eval 3 scenes × 2 IF questions × 6 pts = 36 of 51 points).

## Verdict

The IF pipeline's *planner internals* (A* threading, capsule stamping, breadcrumb emission) are individually careful, but the pipeline as wired will lose most IF points at eval for two compounding reasons: (1) for IF questions the robot performs **no exploration at all** — it publishes zero waypoints until *every* route anchor already has a detection, which an unseen scene will frequently never provide from spawn (verified empirically: 30 ticks against an empty scene → 0 waypoints); and (2) the GT battery measures **planned-path Fréchet/coverage**, while the challenge scores the **driven trajectory on ordered constraint adherence with forbidden-region penalties** — our scorer never checks ordering, gate threading, or avoid violation even though the toolbox contains exactly those checks (`threading_check`, `capsule_violated`), which are only called from tests. The three "unaligned" scenes are best explained as wrong terminal-goal grounding (the 2-point rigid fit can only have residual > 0 when our two resolved goal centroids' separation disagrees with the GT endpoints' separation), i.e. goal disambiguation is confirmed as the suspect — but the measured 6.1 m / 30% numbers are themselves optimistic, produced under full observation, no walls, and a GT-matched spawn that eval never provides.

---

## F1 — For IF questions the robot never moves until every anchor is grounded; there is no IF exploration path at all

**Severity:** Highest — up to the full 36 eval IF points. Any IF question with one anchor not detectable from the spawn area (multi-room scenes, small/distant objects like "tray on the table", occluded anchors) produces a robot that sits still for ~570 s and then receives a single floor waypoint. Score ≈ 0/6 on those questions; `home_building_*`-style scenes make this near-certain.

**Evidence:**
- `src/core/heads/explore_step.py:107-113` — `ExploreHead.advance`: for `QType.INSTRUCTION_FOLLOWING` it delegates to `InstructionHead.advance` and **returns**; `_explore` (orientation diamond sweep + frontier exploration, `ExplorationPolicy`) is never reached for IF.
- `src/core/heads/instruction.py:208-232` — `_build_route` returns without a follower while `any(l.geom is None)`; `_drive` (`:292-303`) publishes nothing without a follower. No waypoint is ever emitted before all anchors resolve.
- Empirical probe (this review): `InstructionHead.advance` driven 30 ticks against an empty `BasicSceneIndex` → **0 waypoints published**, `ungrounded_subgoals=2`.
- The advertised mitigation is unimplemented: `instruction.py:384-389` `next_noun_affinity_target` (the "interleaved explore-execute" bias of architecture §4, `docs/architecture.md:95-98`) is called **only by a test** (`src/tests/heads/test_instruction.py:123`), never by production code. The docstring claim "KEEP driving the current leg" (`instruction.py:15-19`) has no implementation — there is no partial-route drive.
- Compounding: the FSM's ORIENT state does nothing for 60 s (`src/core/fsm/controller.py:203-206` waits for the window; the sweep waypoints live in `ExplorationPolicy.step`, `src/core/nav/exploration.py:104-113`, which for IF never runs). So even the map-seeding sweep never happens for IF questions.
- Fallback at 570 s: `src/core/fsm/floors.py:180-194` publishes one waypoint at the first grounded anchor, else the scene centroid, else the origin.

**Attack scenario:** Test scene shaped like `home_building_2`, question "First, go to the nightstand with a clock on it, then …" with the nightstand in another room. The robot latches the question, idles through ORIENT, then ticks `InstructionHead.advance` for ~8 minutes with `geom=None` legs; publishes its first and only waypoint at T-30 s. Driven trajectory ≈ a point. 0/6, twice per such scene. (The GT battery cannot see this: `_drive_if_path` bypasses the FSM with a fully observed scene — see F2.)

**Proposed fix (design):** Make the IF head drive partial routes: plan and drive through the longest grounded prefix of legs while frontier exploration (noun-affinity-biased, honouring avoid capsules) runs whenever the current leg or any later leg is ungrounded. Run the orientation sweep for IF too. Wire `next_noun_affinity_target` into the explore affinity as designed. Partial credit exists ("path-constraint adherence in correct order, partial credit possible" — `docs/question_analysis.md` §2, quoting the upstream README) — driving the first two grounded legs banks real points even if the terminal anchor is never found.

**Cost:** M (the pieces exist — ExplorationPolicy, affinity hook, plan_through over leg prefixes — this is wiring plus arbitration logic, not new algorithms).

---

## F2 — The GT scorer measures the wrong thing: planned-path Fréchet/coverage instead of driven-trajectory ordered-constraint adherence

**Severity:** Catastrophic-silent — it does not lose points by itself, but it makes every other IF defect invisible or mis-prioritised. The 6.1 m / 30% headline is computed in a regime eval never provides, and improving it does not provably improve challenge score.

**Evidence:**
- Real rubric: the trajectory scored is the **actual followed trajectory** (`docs/upstream_notes.md:185`, quoting upstream `README.md:175`), scored as "path-constraint adherence in correct order, partial credit possible", with explicit penalties for forbidden regions (`docs/question_analysis.md` §2 and §5 "Negative/forbidden-path constraints … scored").
- What we score instead: `src/core/groundtruth/scoring.py:851-918` `score_instruction_following` computes discrete Fréchet + coverage@1m of `our_path` against the reference PLY — and `our_path` is the **planned** `BreadcrumbFollower.path`, not a driven trajectory (`src/core/runner/gt_battery.py:100-152` `_drive_if_path` returns `follower.path`).
- Nothing in the scorer checks ordering, gate threading, or avoid violation. The toolbox has purpose-built oracles — `threading_check` (`src/core/geometry/toolbox.py:573`) and `capsule_violated` (`:625`) — but grep shows they are referenced **only from `src/tests/`**; `gt_battery.py`/`scoring.py` never import them.
- The battery regime is systematically optimistic: fully observed scene (`gt_battery.py:11-13`), spawn placed at the GT trajectory's mapped start (`:283-289`), FSM/watchdog/exploration bypassed (`:130-143`), and the mirror costmap **has no walls** (`_synthetic_from_gt` docstring, `:71-73`) — so in multi-room scenes our planned path can legally cut through where walls are (consistent with `home_building_1`'s Fréchet 22.6 m / 20.1 m rows in `reports/gt_battery_full_2026-07-11/gt_battery_report.md:33-34`).
- Direction of error is unknown per question: Fréchet punishes shape differences the rubric may not care about, while coverage@1m can look healthy on a path that skipped an ordered via or crossed a forbidden corridor — the two things the rubric actually pays/penalises.

**Attack scenario:** We tune the pipeline to lift coverage@1m from 30% to 60% by smoothing planned paths; at eval the driven trajectory (exploration prefix + local-planner deviations + F4's terminal-jump) still misses ordered constraints and clips an avoid corridor. Measured metric doubled; challenge score unchanged or worse.

**Proposed fix (design):** Rebuild the IF scorer around a rubric proxy: simulate the *driven* trajectory (MockRobotIO pose stream under the FSM, not `follower.path`), then score (a) per-leg arrival within a tolerance **in order**, (b) `threading_check` per corridor leg, (c) `capsule_violated` per avoid spec, (d) keep Fréchet/coverage as secondary diagnostics only. Add walls from the VLA-3D region data to the mirror scene, and run a no-spawn-hint variant so exploration cost is visible.

**Cost:** M (all oracles exist; the work is harness plumbing and a scored-rubric aggregate).

---

## F3 — Terminal-goal disambiguation: the 3 unaligned scenes are wrong-instance groundings, and the resolver's tie-breaking is arbitrary

**Severity:** High — a wrong terminal goal converts a 6-pt question into ≤1-2 pts (early legs only). The battery already shows ≥1 gross mis-grounding in 3 of 15 scenes (20%), and eval scenes are unseen.

**Evidence:**
- Geometry of the diagnosis: `align_scene_trajectories` (`scoring.py:774-820`) fits one rigid 2D transform from exactly two correspondences (each GT trajectory endpoint → our resolved terminal-goal centroid, `gt_battery.py:155-185`, `:262-289`). A proper rigid fit of 2 points has residual `|d_src − d_dst|/2` — nonzero **only** when the separation of the two GT endpoints disagrees with the separation of our two resolved goals. `home_building_2` residual 6.02 m ⇒ ~12 m disagreement; `livingroom_3` 1.81 m; `hotel_room_2` 1.09 m (`gt_battery_report.md:38-39,48-49,68-69`). Frames cannot explain this (rotation/translation/reflection all preserve distances); **at least one terminal goal per scene resolved to the wrong instance** (or the parser extracted the wrong terminal anchor).
- Arbitrary salience: `resolve` picks a superlative's anchor as `anchor_recs[0]` — "first (index order); deterministic" (`toolbox.py:495-497`); ambiguous survivors are ordered by `instance_id` (`toolbox.py:538-539`). "Stop at the lamp …" in a hotel room with two identical lamps returns the lowest-id lamp, i.e. detection order, not salience.
- Silent relaxation: the fallback ladder (`toolbox.py:460-490`) drops attributes, then relations, then falls to category-only — so an unsatisfiable disambiguator ("the pillow farthest from the window" with the window undetected) degrades to *some pillow* with no confidence penalty visible to the head; `_ground_one` (`instruction.py:147-168`) treats the result identically to a clean resolve.
- The head commits to `ranked[0]` at first route build and never revisits (see F8); CP3 demote-and-replan exists (`instruction.py:327-357`) but the CP3 seam is None in today's wiring (`factory.py` defaults), so it is inert.

**Attack scenario:** `hotel_room_2`-like eval scene: "Go between the bench and the bed and stop at the lamp closest to the bench." Both nightstand lamps are detected; the disambiguating anchor ("bench") is not yet detected when the route firms up → relaxation ladder drops the clause → category-only → lowest-id lamp on the far side. Robot drives a plausible-looking route to the wrong lamp; terminal constraint scored 0.

**Proposed fix (design):** (1) Treat relaxation-audited groundings as *provisional*: do not commit a terminal leg whose resolve used `drop_relation`/`category_only` while budget remains — keep exploring for the missing disambiguator (ties into F1's partial-route drive). (2) Replace index-order anchor salience with nearest-to-target or largest-instance salience. (3) Wire CP3 anchor confirmation for the terminal leg specifically (cheap: 1 call, cap already in `budget.py:97-104`). (4) In the battery, report the resolve audit trail per IF leg so wrong-instance groundings are countable offline.

**Cost:** M.

---

## F4 — Budget-exhaust handoff: the FSM abandons breadcrumb guidance mid-route and leaves one distant terminal waypoint standing

**Severity:** High — affects every IF question whose route is not finished by the ~270 s soft budget (likely most, once exploration is added). Loses ordering and avoid compliance for the trajectory tail and risks stranding.

**Evidence:**
- IF never early-answers (`controller.py:249-260`), so the IF path to ANSWER is `past_explore_budget` (270 s, `docs/architecture.md:109`) or forced assembly at 510 s. `_tick_verify` → `_tick_answer` publishes the head's `terminal_waypoint()` — the terminal goal coordinate (`factory.py:191-202`, `instruction.py:391-395`) — then the FSM goes DONE and **never ticks the heads again** (`controller.py:161-165`), so breadcrumb emission stops.
- The base stack then drives toward one far waypoint: `waypointConverter` re-projects it each cycle and the stack explicitly warns a distant waypoint can strand the vehicle at a dead end (`docs/upstream_notes.md:456-472`, gotcha 14 — the very reason breadcrumbs exist, `src/core/nav/breadcrumbs.py:1-9`).
- The straight-line dash to the terminal is invisible to our avoid capsules (they exist only in our costmap; the base local planner knows nothing of them — `upstream_notes.md:324-339` shows the `/way_point` plumbing has no such concept).

**Attack scenario:** "First go near A, take the path between B and C, avoiding the path near D, and stop at E." At 270 s the robot is between legs 1 and 2. FSM answers: publishes E's coordinates 9 m away, goes DONE. The vehicle beelines, never threads the B–C gate (ordered constraint lost), passes within D's forbidden disc (penalty), or wedges at a dead end (trajectory ends far from E).

**Proposed fix (design):** For IF, "answering" must mean *continuing the drive*: keep ticking the instruction head (breadcrumbs + capsule-aware planning) until the route completes or the watchdog fires; publish the terminal waypoint only when the follower is exhausted and the vehicle is within reach of the terminal. Reconsider the 270 s soft budget for IF — the drive IS the answer, so answer-path pressure should not cut it off at 45% of the window.

**Cost:** S-M (FSM state semantics change for one qtype; no new machinery).

---

## F5 — Avoid ("do not pass") semantics: capsules stamped once, silently dropped when unresolvable, and never enforced on the driven trajectory

**Severity:** High per affected question — avoid violations are explicitly penalised, and 3 training questions carry avoid clauses (`docs/question_analysis.md` §5); eval scenes likely include ≥1.

**Evidence:**
- `_stamp_avoids` runs once inside `_build_route` and skips any spec whose anchors don't resolve: `except ValueError: continue` (`instruction.py:276-289`, `toolbox.py:588-622`). Because `_build_route` only requires *route-leg* geometry — avoid anchors are not in `route` — the route can firm up while the avoid anchors (e.g. "the TV") are still undetected: capsule never stamped, never retried.
- No runtime monitor: `capsule_violated` (`toolbox.py:625`) is never called outside tests (grep across `src/core`), so a driven violation is silent — no abort, no re-route, no flight-recorder event.
- The base stack can breach the capsule on our behalf: `waypointConverter` snaps an untraversable waypoint to the nearest traversable terrain point (`upstream_notes.md:330-339`) with no knowledge of our capsules, and the local planner deviates around obstacles freely; a crumb near the capsule boundary can be executed inside it.
- Once F1 is fixed, exploration waypoints (`explore_step.py:140-148`) also carry no capsule check — the exploration phase of the very same scored trajectory can cross the forbidden corridor before the route is even planned.

**Attack scenario:** `livingroom_2`-style question, "…avoiding the path between the TV and the tea table." TV undetected at route build (dark panel, low detector score). No capsule stamped. Planned route happens to use the TV–tea-table gap because it is the shortest corridor. Full avoid penalty; our own logs say the route was "capsule-validated".

**Proposed fix (design):** (1) Make avoid-anchor grounding a route-build precondition of the same rank as leg grounding (do not commit a route while an avoid spec is unresolvable; bias exploration toward avoid nouns too). (2) Re-stamp on every re-ground; if a capsule appears after the follower exists, force a re-plan. (3) Add a runtime tripwire: each tick, test the current pose (and next crumb) against all capsules with `capsule_violated`; on entry, stop and re-plan away — plus log the event. (4) Keep crumbs a safety margin away from capsule boundaries (inflate the capsule for crumb selection beyond the vehicle radius already applied at `costmap.py:96-97`).

**Cost:** M.

---

## F6 — "the two X" corridors produce a degenerate zero-width gate and collapse the whole route to the recovery fallback

**Severity:** Medium-high — "path between" appears in 10/30 training IF questions; the counted-pair phrasing ("the two columns") is a live training case (`arabic_room` q5). When it fires, all ordered legs are abandoned for the nearest-legal-point recovery: most of that question's 6 pts.

**Evidence:**
- Parser duplicates the anchor for "the two/both X": `_split_pair` (`src/core/parsing/regex_tier.py:181-192`) returns two `Anchor(noun="column")`s. Verified: parsing arabic_room q5 yields `corridor_between ['column', 'column']`.
- Grounding resolves both anchors independently and both get the SAME `ranked[0]` instance (`instruction.py:151-182` — nothing excludes the first anchor's pick when resolving the second).
- Verified numerically: `corridor_gate(rec, rec)` returns a gate with `p0 == p1`, `width = 0.0` (probe run, this review; `toolbox.py:565-570`).
- Consequences in `plan_through` (`src/core/nav/planner.py:183-223`): the gate midpoint is the object's own face point; `path_crosses_gate` against a zero-length segment is nearly unsatisfiable; the pinch overlay then blocks everything within 3 m except a 0.5 m-half-width corridor around a *point* that sits at the object's face — inside the inflated obstacle (`Costmap` inflation 0.4 m, `costmap.py:24`) — so A* fails and `plan_through` returns None; `_build_route` falls to `_recover_path` (`instruction.py:226-232`), which drives to the nearest reachable point to the terminal and skips every intermediate constraint.

**Attack scenario:** Exactly arabic_room q5 phrasing in an eval scene: "…take the path between the two columns, and stop at the tray on the table." Route → recovery path: no corridor threading, first leg skipped, trajectory ≈ beeline. The flight recorder logs it as a graceful recovery, masking that the cause is a parser/grounding bug, not true unreachability.

**Proposed fix (design):** Pair-instance resolution: when a corridor leg's two anchors share a noun, resolve the top-2 *distinct* instances of that noun (fail to ungrounded if <2 exist). Generally, enforce distinct-instance constraints across a leg's anchors and across BETWEEN clause anchors (same bug shape exists for `avoid between` via `_split_pair`'s duplicate-on-failure fallback, `regex_tier.py:190-192`).

**Cost:** S.

---

## F7 — Corridor and via geometry is plan-time-only and orientation-blind

**Severity:** Medium — silently degrades corridor ("between") and "path near" legs; these are 14 of 30 training IF questions (`path between` 10, `path near` 4).

**Evidence:**
- `_via_point` places the "near" waypoint at centroid + **(+1.2 m, 0)** — a fixed +x offset regardless of walls, approach direction, or object size (`instruction.py:49,189-192`); `_project_free` then snaps to the Euclidean-nearest passable cell (`costmap.py:163-170`), which can be on the far side of a wall or of the object (nearest ≠ reachable-side; `_nearest_passable_cell` ignores connectivity).
- Gate threading is verified only on the planned polyline (`planner.py:207-213`). The driven trajectory differs: `BreadcrumbFollower` emits the farthest LOS point ≤2.5 m ahead (`breadcrumbs.py:91-123`), letting the vehicle cut corners across path vertices near the gate, and `waypointConverter` snapping (`upstream_notes.md:330-339`) can displace the gate-midpoint crumb sideways out of the gap. No runtime `threading_check` confirms the crossing ever happened (grep: tests only).
- A gate midpoint that lands in inflated obstacle (two anchors < ~0.8 m apart + 0.4 m inflation) is snapped by `_snap_passable` (`planner.py:100-105`) to the nearest passable cell — possibly on the near side, making "crossed the gate" false and forcing the pinch path, whose 0.5 m half-width (`planner.py:28`) is only marginally wider than the 0.4 m vehicle inflation: narrow real gaps fail to recovery.

**Attack scenario:** "Take the path near the window to the fridge" with the window on the +x wall: via point lands inside the wall, projects to a cell on whichever side is Euclidean-closest — potentially outside the room. Or a 1.0 m furniture gap: inflation eats the pinch corridor, A* fails, route collapses to recovery (F6's failure mode without the parser bug).

**Proposed fix (design):** Via points: offset along the free-space gradient (pick the passable cell at `near_thresh` of the anchor with maximum clearance on the robot's side of the anchor, reachable from the current pose). Threading: after crossing, run `threading_check` on the odom trail; if missed, insert a corrective loop while budget remains. Gate feasibility: check gate width against vehicle diameter before committing a corridor leg; if too narrow, treat as via_near at the widest adjacent gap.

**Cost:** M.

---

## F8 — The route is frozen at first any-candidate grounding and there is no replanning: `replan_flag` is dead code

**Severity:** Medium — turns early perception noise into committed wrong routes and makes stalls terminal.

**Evidence:**
- `_build_route` fires as soon as every leg has ANY candidate — `geom is not None` requires only one resolve candidate (`n_obs >= 1`); the `grounded` flag (`MIN_GROUND_OBS = 3`, `instruction.py:51,155`) gates nothing in route construction, only the FSM early-answer readout (`instruction.py:369-374`) — which never applies because IF never early-answers (`controller.py:258-259`).
- After the follower exists, `_ground_legs` keeps re-resolving every tick (`instruction.py:120-124`) but the result is discarded for routing: `_build_route` is guarded by `if self._follower is None`. Only the (currently un-wired) CP3 demote path ever rebuilds (`instruction.py:352-357`).
- `BreadcrumbFollower.replan_flag` (stall over 10 s, or no LOS crumb — `breadcrumbs.py:88,119-122,188-194`) is set but **never read by production code** (grep: only `src/tests/nav/test_breadcrumbs.py`). A vehicle wedged by a snapped waypoint republishes the same crumb until the watchdog.

**Attack scenario:** 90 s in, one blurry far-field detection of a "vase" (n_obs=1) gives the terminal leg a geometry; route commits and drives across the scene. At 200 s the true vase (n_obs=5, right room) is mapped; the head re-resolves it every tick, logs nothing, and keeps driving to the phantom. Alternatively: a crumb snaps against a bar table the terrain called free; stall flag raises for 6 minutes; nobody reads it.

**Proposed fix (design):** (1) Rebuild the follower when re-grounding moves any leg's geometry by more than a threshold (with hysteresis to avoid thrash), or at minimum when a leg's resolved `instance_id` changes. (2) Gate route *commitment* (not planning) on `MIN_GROUND_OBS` for the legs already reachable, consistent with the architecture's "grounded == ≥3 obs" claim. (3) Poll `replan_flag` in `_drive`; on stall, re-plan from the current pose (the costmap is already question-scoped).

**Cost:** S-M.

---

## What I could not verify

- **The actual IF scoring function.** The `challenge_evaluation_node` is closed-source (`docs/upstream_notes.md:268-270`, U4). "Path-constraint adherence in correct order, partial credit possible" and "scoring is over the actual followed trajectory" are quoted from the upstream README via `docs/question_analysis.md` §2 and `docs/upstream_notes.md:185`; the upstream clone is not present in this worktree (git-ignored), so I could not re-read `README.md:175` directly. If the organisers actually score trajectory similarity to the reference PLY, F2's direction changes (Fréchet would then be the right family of metric) — but the ordering/avoid language in the README makes constraint-based scoring far more likely.
- **Whether the scored trajectory includes the exploration phase.** Scoring "the actual followed trajectory" strongly suggests the whole drive (making exploration-phase avoid violations, F5, real), but the eval node could plausibly window the trajectory after some event. Unresolvable without the eval node.
- **Which specific instance was mis-grounded in each unaligned scene.** The residual math proves ≥1 wrong terminal goal per unaligned scene but not which question; per-question resolve audit trails are not persisted by the battery (proposed in F3). The report truncates question text, and I did not have the VLA-3D scene folders available in-session to re-run the battery.
- **Real detector behaviour.** All grounding attacks assume plausible detection gaps/noise; the review ran against mocks and the GT index (as the battery does). Magnitudes of F3/F8 depend on the real perception stack's miss/latency profile on unseen Unity scenes.
- **`waypointConverter` snapping displacing crumbs out of gates / into capsules** (F5/F7) is inferred from the upstream notes' description of the snapping mechanism (`upstream_notes.md:330-339,456-458`); I could not execute the base stack on Windows to observe it.
