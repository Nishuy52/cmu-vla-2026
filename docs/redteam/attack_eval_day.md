# Red-team review — eval-day systems failure

**Date:** 12 Jul 2026
**Facet:** ways this system scores zero (or near-zero) on eval day for reasons unrelated to reasoning quality.
**Points frame:** eval = 3 held-out scenes × (1 numerical ×1 + 2 object-ref ×2 + 2 instruction-following ×6) = **51 points total; 36 of them are IF** (docs/challenge_brief.md:20–26, docs/upstream_notes.md §6).

## Verdict

The core FSM/watchdog design is genuinely robust *as a state machine* — floors are total functions, dispatch is type-checked, the watchdog always has a legal object to publish once a question is latched. But the committed eval path around it fails before reasoning ever matters: the adapter ships with a permanently empty scene index (perception is not wired into the node at all), the `/challenge_question` subscription requests a QoS durability the evaluator's publisher almost certainly does not offer (which silences the entire run — the one failure the watchdog cannot save, because the watchdog is downstream of question receipt), and instruction-following questions — 71% of the points — deadlock by construction: the IF head publishes no waypoints until all anchors are grounded, anchors cannot ground without exploring, and the explore head never explores for IF questions. Separately, the LLM failover ladder, every checkpoint seam, and the local-VLM dark-network tier are dead code on the eval path; the budget clock starts later than the evaluator's; and the Docker packaging as written cannot be built the way the evaluators build it. Most fixes are small; the danger is that the current system *passes its own tests* (it always publishes something), so nothing forces these to surface before Ubuntu.

---

## F1 — Perception is not wired into the adapter: every question is answered from an empty scene

**Severity:** ~48/51 points. As committed, the eval container answers every question from `BasicSceneIndex([])`: numerical floor `IntAnswer(2)` (right only when the answer happens to be 2), object-ref floor a 1×1×1 box at the origin (~0 overlap), IF floor `WaypointCmd(0,0)` (robot spawn area). Probability 1.0 if shipped as-is; this is acknowledged Phase-2 work, but it is the single load-bearing integration and nothing gates submission on it.

**Evidence:**
- `src/ros_adapter/adapter_node.py:228` — `self._scene_index = BasicSceneIndex([])` with a "confirm on Ubuntu: swap … for the live perception scene index" comment; `:293` — `build_callables(self._scene_index)` with no other arguments.
- The camera pano is latched (`adapter_node.py:247–253`) and then **never consumed** — no code path in the node feeds `core/perception/` (detector/fusion/tracker exist only under `core/` and the replay runner `src/core/runner/single.py:186–247`, which drives a *scripted* detector, not a real one).
- Floor values: `src/core/fsm/floors.py:31` (`MODAL_COUNT=2`), `:56–60` (origin unit box, `WaypointCmd(0,0)`), `:180–194` (IF floor = instance-centroid mean, which is empty here).
- `docker/ai_module/Dockerfile:28–29` — torch/GroundingDINO install is commented out ("Phase-2: uncomment after weights decision"), so the image cannot run perception even if wired.

**Attack scenario:** eval day, any scene, any question. Node comes up, subscribes cleanly, ticks at 5 Hz, publishes a legal-typed answer before T−30 s on every question. Every smoke test "passes" — and the score is ≈2–4 points of luck out of 51. Silence was avoided; scoring was not.

**Proposed fix:** make perception wiring a named, gated deliverable, not a comment: a `PerceptionPipeline` instance in the adapter fed by pano+scan on new-pano ticks (the exact `_ScriptedPerception.maybe_process` pattern in `runner/single.py:231–247`, with the real detector), with a hard Gate item "instances_tracked > 0 on a live scene" in the phase-2 playbook before any submission. Add a startup log line that shouts when the index is the empty stub.

**Cost:** L (the detector integration itself), Ubuntu-gate — but the wiring seam and the "refuse to submit with stub index" guard are S and can land **Windows-now**.

---

## F2 — `/challenge_question` QoS: TRANSIENT_LOCAL subscriber vs (almost certainly) VOLATILE publisher = the question never arrives = total silence

**Severity:** 51/51 points — the only failure class the watchdog cannot catch, because the watchdog arms only after a question is latched (`controller.py:177` guards on `self.budget`, which exists only after `_intake`; `adapter_node.py:291–292` doesn't even construct the controller until `question()` is non-None). No question ⇒ no budget ⇒ no floor ⇒ nothing on any answer topic for the whole 10 minutes, every question. Probability high (~0.8–0.9): in DDS, a subscriber *requesting* TRANSIENT_LOCAL durability is incompatible with a publisher *offering* VOLATILE — no messages flow, and rclpy only emits a warning log. The dummy subscribes with the default (volatile) profile and works, which strongly implies the closed-source evaluation node publishes with defaults, i.e. VOLATILE.

**Evidence:**
- `src/ros_adapter/adapter_node.py:114–121` (`_reliable_transient_qos`), `:183–185` (question subscription uses it).
- `docs/upstream_notes.md:69–77` — dummy QoS "is just the integer depth (default reliable/volatile profile)"; the question publisher is closed-source (`upstream_notes.md` U4).
- Already flagged as a "runner-up" risk in `docs/tasks/T4-implementation-gaps/task.md` ("if the hidden evaluator publishes VOLATILE, QoS won't match and the module silently gets zero questions") — it is ranked far too low there: this is a whole-run zero, not a footnote.

**Attack scenario:** evaluator launches the stack; question republishes at 1 Hz forever; our subscription never matches; `vla_ai_module` logs "up: subscribed 6 topics" and then does nothing for 600 s. Repeat ×15 questions. Total score 0.

**Proposed fix:** TRANSIENT_LOCAL buys nothing here — the 1 Hz republish *is* the late-join protection (`upstream_notes.md` gotcha 3). Subscribe RELIABLE + VOLATILE (or simply depth-5 default, exactly like the dummy). If durability paranoia remains, create **two** subscriptions (volatile + transient_local) feeding the same latch; matching either wins.

**Cost:** S (one-line QoS change; dual-sub is ~10 lines), **Windows-now**. Verify matched-publisher count against the real stack at the Ubuntu gate (`ros2 topic info -v /challenge_question`).

---

## F3 — Instruction-following deadlock: the IF head never explores, and can never ground without exploring — the robot sits still for 600 s on 36/51 points

**Severity:** ~36/51 points (both IF questions, all 3 scenes, reduced only by whatever partial trajectory credit a floor waypoint at T−30 s earns). Probability ~0.9 *even after F1 is fixed*: grounding requires ≥3 observations of each anchor (`instruction.py:51,155`), observations require motion, and for IF questions no motion is ever commanded until grounding succeeds.

**Evidence:**
- `src/core/heads/explore_step.py:107–113` — `ExploreHead.advance`: for `QType.INSTRUCTION_FOLLOWING` it delegates to `InstructionHead.advance` and **returns**; the frontier/sweep exploration path (`self._explore`) is unreachable for IF.
- `src/core/heads/instruction.py:214–215` — `_build_route` returns unless **every** leg has geometry; `:292–294` — `_drive` returns immediately while `_follower is None`. Net: zero `publish_waypoint` calls until all anchors resolve.
- The advertised mitigation is not wired: "interleaved explore-execute … bias exploration affinity toward the ungrounded noun" (`instruction.py:16–19`, architecture §4) — but `next_noun_affinity_target()` is called **only by a unit test** (grep: `src/tests/heads/test_instruction.py:123`), never by `ExploreHead` or the factory.
- Contrast with the FSM's assumption: `controller.py:258–259` says "only the budget/forced path ends IF" — which presumes IF is *driving* in the meantime. It is not.

**Attack scenario:** IF question, unseen scene. Parse (regex) yields a route with 2–3 anchors. Scene index is empty or sparse at t=0; `_ground_legs` fails every tick; no waypoint is ever published; robot never moves; nothing is ever observed (self-sustaining). At 570 s the watchdog publishes the IF floor — `first_anchor_pt` is None and the scene centroid is a handful of instances visible from the spawn point (or origin) — one waypoint, ≤30 s of driving, ~0–1 of 6 points. ×6 questions.

**Proposed fix (design):** in `ExploreHead.advance`, when qtype is IF, delegate to the instruction head **and then fall through to `_explore`** whenever `instruction._follower is None` (or `ungrounded_subgoals() > 0` and no waypoint was emitted this tick), with the affinity factory fed by `next_noun_affinity_target()`. That is precisely the architecture-§4 row-10 behaviour, and it's testable on Windows today with the existing mocks (a test asserting "IF question + empty scene ⇒ waypoints are still published within N ticks" would have caught this).

**Cost:** M (~50 lines + tests), **Windows-now**. Nothing about it is Ubuntu-blocked.

---

## F4 — Docker packaging cannot be built the way the evaluators build it (context mismatch, PEP 668 pip, unproven colcon layout)

**Severity:** up to 51/51 if a broken image reaches submission (evaluators pull and run; a build/run failure = zero on everything), but realistically caught at the Ubuntu gate — call it moderate probability of *eating days at the gate* and nonzero probability of a submission that runs differently at the evaluator's machine than ours.

**Evidence:**
- `docker/ai_module/Dockerfile:31–33` — "Build context is the repo root … COPY src/ /opt/vla/src/". But the upstream compose builds the ai_module service **from `../ai_module` with `ai_module/docker/Dockerfile`** (`docs/upstream_notes.md:286–291`), and submission rules allow modifying **only `ai_module/`** (`upstream_notes.md:302–306`, gotcha 12). Our Dockerfile lives at `docker/ai_module/` in *our* repo with a root build context — as written it does not slot into the fork's `ai_module/docker/Dockerfile` + `ai_module/`-context shape at all. Someone must restructure at the gate; the restructure is untested.
- `Dockerfile:24–27` — bare `pip install` on an Ubuntu 24.04 (Noble) base: PEP 668 "externally-managed-environment" makes system-pip installs **fail by default** unless the base image pre-configured a venv or `--break-system-packages`. Unverifiable until the base image is pulled; if it fails, the image doesn't build.
- `Dockerfile:40–43` + `src/ros_adapter/setup.py:9–15` — colcon build of an ament_python package via a **symlink** into the workspace, with `package_dir={"ros_adapter": "."}`, plus the "confirm on Ubuntu that `ros2 run vla_ai_module adapter_node` resolves `core`" note. Three separate acknowledged unknowns (ament+core coexistence, scene-index wiring, colcon layout — LOG.md:245) all sit on the only path from `docker run` to a spinning node.
- `Dockerfile:10` — `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` baked via ENV: correct intent (gotcha 13), but ENV is overridden by compose `environment:` blocks and not visible to `docker exec` shells that don't inherit it; a mismatch here is **silent discovery failure** (no topics, no question — same terminal outcome as F2).

**Attack scenario:** Ubuntu day 1: base image pulled, `docker build` fails at the pip layer (PEP 668) or the colcon layer; two days lost. Or it builds locally after hand-restructuring, but the *submitted fork* carries the old layout; evaluators' `docker compose up --build` fails; score 0 with no feedback loop until (if) they contact us.

**Proposed fix:** restructure now, on Windows, to the exact upstream shape: our code under `ai_module/src/` in the fork, Dockerfile at `ai_module/docker/Dockerfile` with `ai_module/` context, mirroring the dummy's build (`upstream_notes.md:286–291`). Add `--break-system-packages` (or a venv) defensively with a comment. Treat "compose up --build from a clean clone of the fork" as the only definition of packaging done.

**Cost:** M, restructure **Windows-now** (it's just file layout + text); validation necessarily **Ubuntu-gate**.

---

## F5 — The dual-API → local-VLM → regex failover ladder is dead code on the eval path; the local tier does not exist at all

**Severity:** not a zero by itself (the regex floor parses *something* and the heads are deterministic), but it silently deletes the entire reasoning tier the architecture budgets for (architecture §3: parse, miss-recovery, anchor-confirm, verification, frontier-select). Expected-points impact concentrates on OR clause verification and IF anchor confirmation — plausibly several points per scene versus design intent. Probability 1.0 as committed.

**Evidence:**
- `src/ros_adapter/adapter_node.py:293` — `build_callables(self._scene_index)` passes **no** `parse`, `llm_verify`, `anchor_confirm`, `verifier`, `miss_recoverer`, `frontier_selector`, `budget_frac`, `remaining_s` — every seam defaults to None/offline.
- `src/core/heads/factory.py:157, 228–233` — default parse is `parse_regex` directly; `ladder.parse` is never in the loop, so the ledger-gated, time-capped API→API2→local chain (`src/core/parsing/ladder.py:36–82`) and the timeout wrapper (`src/core/llm/timeout.py:32` — 20 s/call; `ladder.py:39` — 45 s cap) are exercised only by tests.
- Grep confirms `load_config` / `build_chat_fns` are referenced **only** in `src/core/llm/` and `src/tests/` — no production caller. Even the full runner opts out ("omitted / None keeps the deterministic offline path (regex parse only)", `src/core/runner/single.py:275–276`).
- The "local quantized VLM" dark-network tier has no server, no weights, no process in the image (`Dockerfile:28–29` commented out) — under a dark network the ladder (if it were wired) would burn its API timeouts and land on regex anyway.
- On the positive side, the termination property the brief asks about **does hold structurally**: `ladder.parse` never raises and always returns a Plan (`ladder.py:51, 78–82`); worst-case wall time inside one parse ≈ 80 s (cap checked between attempts, so a repair starting at 44.9 s runs its full 20 s; 3 tiers bound it), which is survivable at t≈0 but see F8 for why it must not run inside the tick thread.

**Proposed fix:** in the adapter (or factory), `parse=lambda q: ladder.parse(q, build_chat_fns(load_config()), clock, ledger)`; wire `remaining_s`/`budget_frac` from `BudgetState`; decide the local-tier reality (ship a llama.cpp/Qwen-VL server in the image, or explicitly descope it and update architecture §3's "fallback if dark" column so the design matches the truth).

**Cost:** wiring S–M **Windows-now** (testable with `LocalStub`); local-VLM tier L, **Ubuntu-gate** (weights + VRAM validation), or a deliberate descope decision now.

---

## F6 — Budget clock starts at question receipt; the evaluator's 10 minutes start at system startup — every gate is later than we think (t_received unit-mismatch lead: resolved, no bug)

**Severity:** overtime penalty / late-answer risk on every question, worst on IF where the floor fires at our-570 s ≈ eval-580–630 s. Probability high that *some* offset exists (Unity boot + `sleep 3` + autonomy launch + DDS discovery + first question publish easily total 10–60 s); severity per question is the README's overrun penalty (magnitude unknown, closed-source scorer) plus losing the early-finish tiebreak.

**Evidence:**
- `docs/upstream_notes.md:437–438` (gotcha 2): "10-minute hard budget per question, **timed from system startup**".
- `src/core/fsm/controller.py:150–152` — `budget.latch(getattr(q, "t_received", None))`: t0 = the *node-clock time our subscription received the first question*, which is strictly later than system startup by the whole boot+discovery pipeline.
- **The nanoseconds-vs-seconds lead is clean:** `adapter_node.py:283` passes `self._now_ns()` (ns) into `string_to_question`, which converts `float(bag_ns) * 1e-9` to seconds (`src/core/replay/bag_reader.py:211–213`); `_NodeClock.now()` is also seconds (`adapter_node.py:140–141`). Same clock, same unit — the watchdog arithmetic is internally consistent. The residual clock hazard is different: if `use_sim_time` were ever set true without a `/clock` source, the node clock freezes and `elapsed()` pins below 60 s ⇒ ORIENT forever + watchdog never fires ⇒ silence (`budget.py:69–71`, `controller.py:205–206`). It is currently unset (wall clock), which is correct — assert it stays that way.
- Aggravator: nothing publishes an early answer to hedge — the first legal answer of any kind goes out at earliest when VERIFY produces one, typically after the per-type explore budget (210–270 s).

**Attack scenario:** slow scene load ⇒ question arrives 45 s after startup. Our watchdog fires at our-570 = eval-615 s. The scorer either rejects the late answer or applies the overrun penalty — on all 15 questions, systematically.

**Proposed fix:** treat the budget as adversarial: (1) drop the gates to 480/540 s (or make them launch params) so the whole floor path clears even with a 60 s startup skew; (2) at the Ubuntu gate, measure real skew (timestamp of first `/state_estimation` vs first `/challenge_question` vs container start) and set the margin from data; (3) publish a provisional floor answer early (e.g. at explore-budget expiry) for numerical/OR, where a later better answer can overwrite it — **only if** the Ubuntu-gate check confirms the scorer reads the *last* value rather than the first (U4 is closed-source; verify empirically with the dummy).

**Cost:** S (constants/params) **Windows-now**; skew measurement + last-vs-first-answer semantics **Ubuntu-gate**.

---

## F7 — Controller qtype is frozen at a regex guess and never corrected by the parse: the watchdog can publish the wrong *type* of answer

**Severity:** whole-question zero whenever the heuristic misfires and the run degrades to the floor path (exactly the runs where the floor matters). OR questions containing motion verbs ("Find the picture you walk past", "…that you pass by the door") classify as IF (`controller.py:312–321` matches `pass|walk|go|move|…`) ⇒ at 570 s the watchdog publishes a **WaypointCmd** while the scorer awaits a Marker on `/selected_object_marker` ⇒ 0/2. Probability: low-moderate per question (training-set phrasings mostly fit the heuristic; held-out phrasing is unknown), but the cost is a categorical zero and the fix is one line.

**Evidence:**
- `src/core/fsm/controller.py:149` — `self.qtype = _infer_qtype(q)` at intake; no code path ever updates `self.qtype` after `self.plan` (which carries the authoritative `plan.qtype` from the parse, `factory.py:77–96`) lands.
- Floor selection keys off the stale guess: `controller.py:228, 284–285` — `self.floors.get(self.qtype)`.
- Budget gates also key off it (`budget.py:73–82`): a mis-typed question additionally gets the wrong explore budget (e.g. 270 s instead of 240 s).

**Attack scenario:** held-out OR question phrased with an embedded motion clause; regex tier parses fine (plan.qtype correct), but detection is weak, verify returns None, floor path fires — with the IF floor. Wrong topic, wrong type, 0 points despite the pipeline holding a valid marker candidate in `partial.best_marker`.

**Proposed fix:** in `_tick_parsing`, after a successful parse: `if isinstance(plan.qtype, QType) and plan.qtype is not self.qtype: self.qtype = plan.qtype; self.budget.qtype = plan.qtype` (log the correction). Keep the heuristic strictly as a pre-parse seed, as its own docstring claims it is (`controller.py:324–328`).

**Cost:** S, **Windows-now**.

---

## F8 — All checkpoint work runs synchronously inside the 5 Hz timer callback: a slow call stalls sensors, ticks, and the watchdog itself

**Severity:** watchdog latency up to ~tens of seconds exactly when it matters (the forced-assembly window), plus sensor-latch starvation during any long call. Combined with F6's late t0 this converts "floor at 570 s" into "floor after 600 s". Probability: certain mechanism, magnitude depends on what gets wired per F5 (today, with everything offline, ticks are fast — the bomb is armed by fixing F5).

**Evidence:**
- `adapter_node.py:232, 287–298` — a single `create_timer` drives `QuestionController.tick(self)`; `main()` uses plain `rclpy.spin` (`:458–462`) = **single-threaded executor**: while `tick` runs, *no* subscription callbacks and no next timer tick execute.
- `controller.py:194–197, 218–220` — parse and verify are called inline in tick; the watchdog overlay is only evaluated at the top of the *next* tick (`controller.py:177–179`).
- Ladder worst case inside one tick ≈ 80 s (F5 evidence); the verify seam (`object_ref.verify` → injected `llm_verify`) has **no** `with_timeout` guarantee at the factory level — the wrapper exists only if whoever wires it remembers (`factory.py:59, 84–91` pass the raw callable through).
- The 45 s `LEDGER_RESERVE_S` (`budget.py:23,143`) assumes a call admitted at remaining=45.1 s finishes before T0 — true only if every wired callable is timeout-bounded well under 45 s *and* the executor can actually run the watchdog tick afterwards.

**Attack scenario:** post-F5 wiring, network degraded (not dark — worse: slow). Forced assembly at 510 s enters VERIFY; `ledger.allow("verification")` passes at 553 s; the API call crawls to its 20 s timeout, then the repair/second-provider path adds more; the next tick begins at 595 s; watchdog publishes at 595–600+ s against an evaluator clock that started earlier (F6). Zero or penalty.

**Proposed fix (design):** (1) move LLM checkpoints off the tick thread — a worker thread + future the FSM polls (tick stays O(ms); states gain a "waiting on checkpoint" no-op branch); or, minimally, (2) `MultiThreadedExecutor` + reentrant callback group so sensor callbacks and the next tick survive a blocked tick, and enforce `with_timeout` on **every** injected seam in `build_callables` itself rather than trusting call sites; (3) shrink the admission bound: require `remaining > reserve + per_call_timeout` in `CallLedger.allow`.

**Cost:** M, **Windows-now** for the FSM-side polling structure and ledger bound (pure Python, fully testable with FakeClock); executor choice validated **Ubuntu-gate**.

---

## F9 — Exploration publishes raw, far frontier centroids (no planner, no breadcrumbs) and can anchor its sweep at (0,0)/t=0 garbage

**Severity:** stuck-robot and wasted-budget modes on numerical/OR (and on IF once F3 is fixed, since exploration becomes IF's bootstrap). Gotcha 14 is explicit: a distant waypoint can strand the vehicle at a dead end. Probability moderate per scene; cost is minutes of the 600 s and sometimes the whole run's coverage.

**Evidence:**
- `src/core/heads/explore_step.py:144–148` — the frontier decision's waypoint (`WaypointCmd(f.xy…)`, a centroid possibly 10+ m away across unexplored space) is published **directly**; `astar`/`BreadcrumbFollower` (`core/nav/planner.py`, `breadcrumbs.py` — built for exactly this, ≤2.5 m LOS-checked crumbs) are wired only into the IF head. `docs/upstream_notes.md:470–472` (gotcha 14) + `interfaces.py:76–81` ("Keep targets near the vehicle (<= ~2.5 m ahead)").
- `explore_step.py:115–118, 133–134` — if the first explore tick runs before the first odom message, `pose=(0,0)` and `t=0.0` are used to construct `ExplorationPolicy(start_xy=(0,0))` and anchor `t0=0.0` (`exploration.py:80–85`). Next tick, `t` jumps to the real odom header stamp; if that stamp is epoch-like/large, `elapsed` is astronomically past `sweep_s` and the 60 s orientation sweep is **skipped entirely** (`exploration.py:105`); the diamond additionally circles (0,0) rather than the spawn point.
- Unreachable frontiers are "heavily deprioritised but still surfaced" with pd=1e6 (`frontiers.py:162–165`) — with a large size term they can still win and be commanded.

**Attack scenario:** relaunch; question latches before the first `/state_estimation` arrives (odom is 100–200 Hz but discovery ordering is not guaranteed); policy anchors at (0,0)/t=0; sweep skipped, so the map is seeded from a single yaw; first frontier centroid is beyond a doorway the local planner dead-ends at; robot parks; frontier list never changes (map static while stationary — waypointConverter snapping keeps it "trying"); explore budget burns to 210–270 s with 10% coverage.

**Proposed fix:** (1) do not construct the policy (or even run `_explore`) until `latest_odom()` is non-None — one guard; (2) route frontier goals through the same `astar` + `BreadcrumbFollower` machinery the IF head uses, or at minimum clamp the published goal to ≤2.5 m along the straight line toward the frontier; (3) drop or hard-cap pd=1e6 frontiers.

**Cost:** (1) S, (2) M, (3) S — all **Windows-now**.

---

## F10 — There is no replanning anywhere: `replan_flag` is never read, the costmap is a one-shot snapshot, a stalled IF drive parks forever

**Severity:** IF partial-credit loss up to whole-question zeros in pinch corridors and cluttered unseen scenes; medium-high probability across 6 IF questions given unseen geometry. (The brief's "capsule-blocked recovery" works at *plan* time — `_recover_path`, `instruction.py:234–267`, is genuinely capsule-safe — but nothing recovers at *drive* time.)

**Evidence:**
- `src/core/nav/breadcrumbs.py:88, 121, 194` — `replan_flag` is set on stall (<0.3 m over 10 s) and on no-LOS-crumb; grep shows the only readers are `breadcrumbs.py` itself and `src/tests/nav/test_breadcrumbs.py`. `InstructionHead._drive` (`instruction.py:292–303`) never consults it and keeps republishing the same crumb.
- `src/core/nav/costmap.py:50–56` — `base_blocked` is computed **once** from `grid.state` at construction; `_build_route` constructs it once per question (`instruction.py:216`). Terrain discovered after route build (new obstacle mid-corridor, an overhead flag appearing under a table) never enters the mask; `line_of_sight` then validates against stale data (breadcrumbs' out-of-bounds conservatism, `breadcrumbs.py:33–46`, covers grid *growth* but not *content* change).
- Route exhausted ⇒ hold terminal goal forever (`instruction.py:298–301`) even if the vehicle physically never got there (arrival marking is distance-based and independent of follower progress, `:305–312`).

**Attack scenario:** IF route threads a corridor gate; base local planner refuses the snapped waypoint behind a just-discovered obstacle; vehicle oscillates <0.3 m; `replan_flag` flips true at t+10 s and is ignored for the remaining 300 s; forced assembly publishes `terminal_waypoint()` — a point the robot never reached; trajectory score ~1/6.

**Proposed fix:** consume the flag: on `replan_flag`, rebuild `Costmap` from the current grid, re-stamp capsules (they are per-question invariants and re-stampable — `_stamp_avoids` is idempotent over specs), `plan_through` again from the current pose, cap replans (e.g. 3/question) and log each to the flight recorder. Same trigger should fire on `at_goal() == False` + follower exhaustion.

**Cost:** M, **Windows-now** (the mocks support stall scenarios today).

---

## F11 — CP2 provisional waypoint latches forever the moment it fires (dormant today, armed by F5's wiring)

**Severity:** one bad VLM hint converts the rest of the question into "drive at a fixed point": exploration stops permanently. Today `miss_recoverer=None` in the adapter so this cannot fire; it becomes live the day the checkpoints are wired — a regression waiting inside a fix.

**Evidence:** `src/core/heads/explore_step.py:139–141` — `if self.provisional_xy is not None: io.publish_waypoint(...); return` on **every** subsequent tick; nothing ever clears `provisional_xy` (no arrival check, no timeout, no observation-confirmation); `_maybe_recover_miss` (`:185–204`) sets it once per question. The design note says the provisional "biases navigation" (`:305–313`) — as coded it *replaces* navigation.

**Attack scenario:** CP2 fires at 60% budget with a frontier-centroid guess inside an unreachable alcove; robot reaches the snapped boundary and sits; the remaining ~2 minutes of explore budget produce zero coverage; count/marker answered from the starved map.

**Proposed fix:** clear `provisional_xy` on (a) arrival within tolerance, (b) T seconds elapsed (e.g. 45 s), or (c) the noun appearing in the scene index; then resume frontier flow. Keep the once-per-question guard.

**Cost:** S, **Windows-now**.

---

## F12 — Overhead-clearance layer: five tunables fitted to one bag, applied asymmetrically (blocks IF routes, invisible to exploration), on unseen scenes

**Severity:** two opposite failure modes on held-out scenes, mostly IF-points: (a) false overhead ⇒ inflated hard blocks sealing doorways ⇒ unreachable routes ⇒ `_recover_path` "answer from nearest legal point" (points quietly lost, by design); (b) missed overhead ⇒ the original under-furniture stranding returns. Probability moderate — this is exactly the "wrong on unseen scenes?" class the brief asks about.

**Evidence:**
- `src/core/nav/occupancy.py:54–68` — `OverheadConfig`: `overhead_min=0.25`, `overhead_max=1.20`, `min_points_per_cell=3`, `vehicle_sensor_height=0.60` (explicitly "jingfan fixtures" calibrated), plus `OVERHEAD_SCAN_MAX_PTS=12000`.
- Interaction bug class: decimation (`occupancy.py:302–335`) strides the scan up to ~5× *before* the ≥3-points-per-cell noise gate — the two tunables fight; a sparse bar-table edge that passes at full density fails after decimation. Neither is validated off the single calibration bag (docs/calibration.md is the sweep record; LOG.md 00482b4 "real-data validated" = one dataset).
- Asymmetry: `Costmap` treats overhead as hard obstacle + 0.4 m inflation (`costmap.py:50–56`) — but only IF routes use a Costmap; exploration publishes raw frontier goals (F9) that ignore `overhead` entirely, and the *base* local planner (which actually drives) reads only the terrain slab (`upstream_notes.md:130–159`), so nothing stops the vehicle physically driving under the very table the IF planner refuses to route under. The layer constrains the half of the system that was already cautious.
- Wall-mounted geometry: anything in the 0.25–1.20 m band over FREE floor (sconces, wall shelves, window sills protruding, curtains) flags cells that then get 0.4 m inflation — adjacent to a doorway this can close the only route (`_recover_path` fires, `instruction.py:234`).

**Proposed fix:** (1) validate the 5 tunables across ≥3 training scenes' bags before trusting them (the fixtures exist for several scenes under `src/core/parsing/fixtures/` naming — record bags at the Ubuntu gate); (2) scale `min_points_per_cell` by the applied decimation stride; (3) make overhead **soft** (high cost) rather than hard-blocked, except in the corridor-threading check; (4) feed overhead into exploration goal filtering too, or accept and document the asymmetry.

**Cost:** (2)(3) S–M **Windows-now**; (1)(4 validation) **Ubuntu-gate**.

---

## F13 — Untimed pure-Python hot loop: frontier clustering/BFS/inflation at 5 Hz plus 100–200 Hz odom callbacks, never measured off the FakeClock

**Severity:** tick overruns ⇒ timer backlog ⇒ everything in F8 gets worse for free; plus stale sensor latches. Probability moderate on the i9 NUC — the numbers are borderline, not obviously safe.

**Evidence:** per explore tick: `integrate_patch` (numpy, fine) + `integrate_scan_overhead_decimated` (12k pts, fine) + `detect_frontiers` = python-loop connected components over the whole mask (`frontiers.py:68–89`) + BFS over all FREE cells (`frontiers.py:92–122`) — a 20×20 m scene at 0.10 m cells is 40k cells; python BFS at ~1–2 µs/op is 50–150 ms *per tick*, against a 200 ms budget, before `Costmap._inflate`'s python disc loop (`costmap.py:59–79`) and `_explored_regions`' full-grid BFS (`explore_step.py:268–302`, runs when CP5 is wired). Meanwhile `_on_odom` runs quaternion math in Python at 100–200 Hz (`adapter_node.py:270–273`, `bag_reader.py:189–208`) on the same executor thread. All timing to date is FakeClock/simulated (`runner/single.py` docstring: "completes in far under a second of wall time").

**Proposed fix:** benchmark on Windows now with a realistic synthetic grid (200×200, 30% free) — it's pure Python, the numbers transfer; convert `_cluster`/`_bfs_distances` to `scipy.ndimage.label`/vectorised BFS if >50 ms; throttle frontier detection to 1 Hz (nothing about frontiers needs 5 Hz); decimate odom conversion (latch raw msg, convert on read).

**Cost:** S (benchmark) **Windows-now**; M (vectorisation) Windows-now; final timing check **Ubuntu-gate**.

---

## F14 — Topic contract and relaunch assumptions: mostly clean, three residual edges

**Severity:** low individually; listed for completeness because the brief demands the sweep.

**Evidence / edges:**
1. **Contract compliance is otherwise verified-good:** the adapter subscribes exactly six legal inputs and publishes exactly the three answer topics (`adapter_node.py:167–196`); nothing publishes `/way_point` directly (gotcha 11); marker uses the leading-slash name (gotcha 5); `theta=0` (gotcha 4); no `/clock`, no TF, no params read from disallowed sources. The debug publishers (`/ai_module/instance_map`, `/ai_module/planned_path`) are created **only** when `debug_viz` is true (`adapter_node.py:203–221`), default false in code *and* in the eval launch (`launch/ai_module.launch.py:28–33`); the eval path allocates nothing. Residual risk is human: the Dockerfile CMD hardcodes the eval launch (good, `Dockerfile:62`) but compose "overrides this with the launch command" (comment, same line) — the override string at the Ubuntu gate must be eyeball-checked for `ai_module_debug`.
2. **Second-question lockout:** `_on_question` latches the first non-empty string forever (`adapter_node.py:275–284`) and the controller ignores different text (`controller.py:155–157`). Correct under relaunch-per-question (gotcha 1) — but if the evaluator's harness ever reuses a running stack (their swap script is not public, upstream U3 residual), we score exactly one question per boot and silently drop the rest. A `docker restart`-tolerant design costs little: on *different* non-empty text, log loudly; optionally exit(0) so the container's restart policy gives the new question a fresh process — decide deliberately rather than by default.
3. **State that must not persist:** nothing in-process persists (fresh process per launch) and nothing is written to disk — verified by inspection (only writers in `core/` are test/fixture paths). The container *filesystem* would persist across in-place restarts; keep it that way (no map caching ever, it's a rules violation per gotcha 1).

**Proposed fix:** (1) add a gate checklist line "grep compose/launch override for debug"; (2) loud-log + deliberate policy on new question text. **Cost:** S, Windows-now.

---

## What I could not verify

### Unverifiable until Ubuntu (needs the real stack / container / hardware)

- **The evaluator's actual QoS profile on `/challenge_question`** (F2) — closed-source publisher; verify with `ros2 topic info -v` against the real stack. Everything about F2 short of that is inference from the dummy's defaults.
- **Whether the closed-source `challenge_evaluation_node` reads the first or last message on each answer topic**, its late-answer/overrun handling, and the true budget origin (F6) — `upstream_notes.md` U4 is permanently closed-source; only empirical probing with the dummy is possible.
- **Base-image realities** (F4): whether `zhangjicmu/ubuntu24_ros:ai_module` permits bare `pip install` (PEP 668), has `/opt/ros/jazzy`, and tolerates the colcon-symlink ament_python layout; whether `ros2 run vla_ai_module adapter_node` resolves `core` (the three in-repo "confirm on Ubuntu" flags: ament+core coexistence, scene-index wiring, colcon layout).
- **CycloneDDS discovery between the two containers** and the Unity TCP bridge on :10000 (whether topics appear at all, and how long discovery takes — feeds F6's skew number).
- **`/camera/image` encoding as republished by `sim_image_repub`** — `image_to_pano` raises on anything outside rgb8/bgr8/rgba8/bgra8 (`bag_reader.py:98–103`); an exotic encoding would silently kill perception (caught by `_safe_probe`, so no crash — just an empty world).
- **Odometry header-stamp epoch in the live sim** (F9's t0 anchor severity) and whether the question can genuinely arrive before the first odom.
- **`waypointConverter` snapping behaviour with our far frontier goals** (does snapping rescue F9 or amplify it), and `/way_point_reached` semantics (breadcrumbs support the signal but nothing feeds it).
- **Real perception latency and the whole F13 timing story on the i9/4090** — Windows benchmarks transfer for pure Python, but executor contention with real 10 Hz images does not.
- **Overhead-layer tunables on any scene other than the jingfan bag** (F12) — needs recorded bags from more training scenes.
- **Whether internet exists on the eval machine** (F5's dark-network question) — the rules allow API tokens, but nothing confirms outbound connectivity from the eval NUC.

### Unverified but checkable now, on Windows (I did not run these; the file is my only write and time was prioritized for it)

- **A live repro of the F3 IF deadlock** via `run_question` with an IF fixture and an empty/sparse index, asserting zero `io.waypoints` before the watchdog — the code path is unambiguous from source (`explore_step.py:107–113`, `instruction.py:214–215, 292–294`), but a failing test should be written anyway as the fix's acceptance test.
- **Regex-tier coverage across all 75 training questions** (F5/F7): what fraction of `questions.json` parses into a usable Plan with qtype matching the bucket — pure-Python, fixtures already in `src/core/parsing/fixtures/`.
- **`_infer_qtype` misfire rate on the 75 training questions** (F7) — same harness.
- **The F13 benchmark** (synthetic 200×200 grid; time `detect_frontiers`, `Costmap.__init__`, `_explored_regions`).
- **Ladder worst-case wall time** under a stub that sleeps to its timeout (confirm the ~80 s bound and the `_expired`-before-repair edge, `ladder.py:70–74`).
- **Floor totality fuzzing:** `FloorAnswers.update/get` under adversarial scene/plan/partial shapes (claimed never-raises; `floors.py:117–129` catches, but `get` on a non-QType, e.g. a raw string, silently returns the IF floor — worth pinning with a test given F7).
- Whether any test currently exercises the adapter's exact `build_callables(scene_index)`-with-no-seams configuration end-to-end (the eval configuration appears to be the one configuration the integration tests never run — they all pass populated indices).
