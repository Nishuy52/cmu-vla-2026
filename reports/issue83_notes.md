# Issue #83 — live exploration starvation, scene-conditional

## Phase 1 — instrumentation (done)

Added `src/core/heads/explore_debug.py` (env-gated on `VLA_EXPLORE_DEBUG_DIR`,
unset = no-op) and wired one call into `ExploreHead._explore` (`src/core/heads/explore_step.py`),
throttled to ~10s of question-clock time. Each JSONL line records:
- costmap summary: grid shape, FREE/UNKNOWN/OBSTACLE cell counts, BFS-reachable
  "pocket" size (cells + m^2) from the current pose
- every clustered frontier blob (pre-filter) with accepted/reason
  (too_small_cluster / unreachable / score_below_min / accepted)
- the waypoint actually published this tick (or null)

Verified: fast test tier green (`pytest -q` from `src/`, 0 failures), manual
smoke test with a synthetic terrain patch confirms JSONL output shape and that
with the env var unset the code path is a single dict-lookup no-op.

**Architecture note found while instrumenting:** the top-level `ExploreHead`
does NOT own the occupancy grid actually driving navigation for
INSTRUCTION_FOLLOWING questions — `InstructionHead` (`src/core/heads/instruction.py:203`)
builds and owns its own separate `OccupancyGrid` for route-following, and only
falls through to `ExploreHead._explore` (and its grid/frontier logic) on ticks
where the route head did not emit a waypoint (no grounded prefix yet). The
instrumentation added here watches `ExploreHead`'s grid/frontier machinery,
which is exactly the mechanism issue #83 names (frontier selection never
firing), and it does get exercised on IF questions once the route is
grounded-yet-idle or before any grounding — confirmed live (see below), but a
future reader should know "the IF costmap" is not a single object.

## Phase 2 — live comparison: office_1 vs livingroom_1 (PARTIAL — stopped early, see below)

### Environment setup (first attempt hit two infra bugs, now fixed)

1. `docker start iros2026_system` alone is not enough after a scene swap +
   container restart: **`xhost +` must be run on the HOST before the
   restart**, not just documented as a step — I initially skipped it on a
   `docker restart` (not a fresh `compose up`, so I assumed the baked
   `DISPLAY=:1` env was sufficient) and got a silent, total sensor-pipeline
   failure: RVIZ's Qt xcb init failed ("Authorization required, but no
   authorization protocol specified" / "could not connect to display :1"),
   `waypointConverter` segfaulted (exit -11) right after, and `/camera/image`,
   `/terrain_map` produced **zero** messages for the life of that run while
   `/parameter_events` etc. still worked — this LOOKS exactly like the
   issue #83 symptom (dead costmap, robot frozen at spawn) but is a **display
   auth infra bug, not the software defect** — a false positive I nearly
   reported as a live repro. Confirmed via the instrumentation itself: the
   first attempted office_1 run dumped `free=1, unknown=31328,
   reachable_pocket_cells=1` unchanged for the full ~95s capture with pose
   pinned at `[0.0, 0.0]` — a *complete* freeze, not the partial-starvation
   pattern the issue describes.
   **Fix:** `DISPLAY=:1 xhost +` on the host before restarting/recreating the
   sim container. `tools/scene_swap.sh` (scratchpad copy) now checks the
   baked `DISPLAY` against `who` and warns, but does NOT re-run `xhost +`
   automatically — that's a manual host-side step per `docs/ubuntu_setup.md`
   §7a's own instructions, this session just failed to do it before the
   first swap. **Follow-up for next session: add `DISPLAY=:N xhost +` as an
   explicit step in `docs/ubuntu_setup.md` §5/§7a's scene-swap sequence,
   not just the initial bring-up sequence** — it's easy to skip on a
   `docker restart` because the DISPLAY env is already baked in and looks
   sufficient.
2. `/challenge_question` publish needs explicit
   `--qos-durability transient_local --qos-reliability reliable` on
   `ros2 topic pub` — a bare publish is VOLATILE by default and the adapter's
   primary subscription is RELIABLE+TRANSIENT_LOCAL
   (`src/ros_adapter/adapter_node.py:199-215`); mismatched durability logs a
   QoS-incompatibility warning and (depending on timing/secondary sub) may
   silently drop the question. `live_question.sh` (scratchpad) sets both
   flags explicitly.

### office_1 / inst — captured, looks HEALTHY (not the starved pattern)

Question: "Go to the potted plant furthest from the projector screen then
stop at the water cooler near the window." Captured
`reports/issue83_live_captures/office_1_inst/explore_debug.jsonl` (6 samples,
~169s of question-clock time, sim restarted clean with `xhost +` applied
first).

| t (s from latch) | status | pose | free cells | reachable pocket (m2) | chosen waypoint |
|---|---|---|---|---|---|
| 0 | sweeping | [3.54, 0.32] | 866 | 0.01 | [4.04, 0.32] (sweep) |
| 10 | sweeping | [3.33, 0.21] | 868 | 6.99 | [4.04, 0.32] (sweep) |
| 75 | frontier | [3.74, 0.36] | 869 | 7.00 | [1.75, 0.45] (real frontier, size 114, score 112.5) |
| 97 | frontier | [3.74, 0.36] | 1082 | 10.19 | [2.79, -1.96] (clamped toward size-156 frontier) |
| 158 | frontier | [3.96, 0.43] | 1212 | 11.46 | [2.86, -1.82] |
| 169 | frontier | [3.85, 0.38] | 1298 | 12.39 | [2.82, -1.90] |

Reads exactly as designed: the opening sweep runs to ~75s (matches the
documented 60s `SWEEP_S` plus orient-state overhead), then `status` flips to
`frontier`, multiple frontier clusters are accepted (not unreachable, not
below the score bar) each tick, the reachable pocket and FREE count grow
monotonically tick over tick, and the chosen waypoint moves to real frontier
goals (clamped to <=2.5m per `FRONTIER_GOAL_CLAMP_M`) rather than staying
diamond-sweep-scale. This does NOT reproduce the issue #83 symptom for
office_1 in this run — consistent with the existing baseline
(`reports/live_baseline_2026-07-20/scores.md`: office_1/inst scored 0.50,
legs=1/2).

### livingroom_1 / inst — ATTEMPTED, TAINTED BY A CONFIRMED GPU WEDGE — inconclusive

Second attempt (after the office_1 capture and a first, self-clearing GPU
blip that turned out to be a stale leftover process, see below). Scene swap
to `livingroom_1` with `xhost +` applied first, sim relaunched clean, host
adapter up with `VLA_EXPLORE_DEBUG_DIR` pointed at
`reports/issue83_live_captures/livingroom_1_inst/`. Question: "Go to the
potted plant closest to the pyramid candle holder and stop at the vase
between the TV and the door." Question latched fine, sweep ran, orient
window closed at 61s (comparable to office_1's ~75s) with the same 5.0Hz
tick rate, GDINO producing real detections early on (GPU healthy under load
at latch time: `P0, ~2000-2500MHz`).

Then, starting ~30s into the explore_execute phase, `nvidia-smi` showed a
**sustained** `210MHz / P8 / 100% utilization / ~9W` reading for the full
remainder of the capture (60+ consecutive seconds, unlike the earlier
office_1-adjacent blip which was a single post-hoc sample). Confirmed with a
direct compute probe (50x 4096x4096 matmuls on the host venv's torch/CUDA):
**34.2s**, vs ~1.0s on a verified-healthy GPU (same probe, same code, run
earlier in this session) — a genuine ~34x throughput collapse, not a
sampling artifact. The debug dump stayed at exactly 1 record (the pre-sweep
sample, pose frozen at `[0.0, 0.0]`, `free=327` unchanged) for the entire
wedge window — i.e. **no further exploration progress was observed while the
GPU was wedged**, which is the same "everything froze" signature as the
earlier xhost-bug false-positive, but this time for a different, real cause
(GPU compute starvation, not a dead sensor pipeline).

Root cause of THIS instance of the wedge: a leftover host adapter process
(GDINO/torch CUDA context) from the run was not actually killed by an
earlier `pkill` (a chained `&&` command aborted before reaching the kill
step because an upstream `pkill` returned nonzero when nothing matched —
classic pkill/`&&` trap). `nvidia-smi --query-compute-apps` found it still
holding 3.5GB of GPU memory and pegging the GPU minutes later. Killing it by
PID dropped utilization to 0% immediately, but a follow-up compute probe
still ran ~9x slower than healthy (8.95s vs ~1.0s) even with 0% reported
utilization and no compute processes listed — i.e. the clock/power-state
itself did not fully recover just from killing the process, consistent with
the standing memory note's "reboot only fix" characterization rather than a
simple process-level fix.

**This capture is INVALID/inconclusive** — the frozen single-sample pattern
cannot be attributed to the software exploration logic; it coincides with a
confirmed hardware-level compute wedge. `reports/issue83_live_captures/livingroom_1_inst/`
is intentionally left EMPTY (not populated with the 1-sample tainted dump) so
nothing here is mistaken for a real repro.

**This is the actual money comparison the issue asks for and it remains
open.** office_1 alone does not tell us why livingroom starves; a valid
same-procedure livingroom_1 capture (scene swap, `xhost +`, sim up, host node
with `VLA_EXPLORE_DEBUG_DIR` set, publish the livingroom_1/inst question with
correct QoS, capture 120-180s, **with per-tick GPU clock/pstate logging and
a pre-flight compute-probe health check**) is the next thing to run, on a
freshly rebooted machine.

### GPU wedge timeline (why Phase 2 ends without a valid livingroom_1 capture)

`nvidia-smi` showed `clocks.sm=210MHz, pstate=P8, utilization=100%,
power.draw~8-9W` at multiple points this session — the signature in this
workspace's standing memory note ("GPU wedge needs reboot... 4060 Laptop can
lock at P8/210MHz under SW power cap; reboot only fix"). Full timeline:
- ~85s into the FIRST (xhost-broken, already-discarded) office_1 attempt:
  healthy (`P0, 2340MHz`). Irrelevant to the wedge question since that run
  was independently invalid (display-auth bug, zero sensor data).
- Checked right after the SECOND office_1 run (the valid capture above,
  no GPU sample taken *during* it): wedged (`P8, 210MHz, 100%util, 7.4W`).
  After killing the host adapter process and stopping the sim container,
  `nvidia-smi` read healthy again (`P0, ~2070-2085MHz`) within about a
  minute, and a direct compute probe (matmul benchmark) confirmed full
  speed (~1.0s/50 matmuls) — this instance cleared on its own once the
  process was gone, so the office_1 capture's data is not suspect (it
  finished before this reading) and live work was judged safe to resume.
- During the livingroom_1 attempt: wedged again, this time SUSTAINED (60+
  consecutive seconds, confirmed via a direct compute probe showing a real
  34x throughput collapse). Root-caused to a leftover host adapter process
  that an earlier `pkill` had failed to actually kill (a chained `&&`
  command silently aborted before reaching the kill step). Killing it by
  PID dropped reported utilization to 0%, but a follow-up compute probe
  still ran ~9x slower than healthy (8.95s vs ~1.0s) — the clock/power
  state did not fully recover just from ending the process, consistent
  with the "reboot only fix" characterization. **This is why the
  livingroom_1 capture is invalid and why Phase 2/3 stop here without a
  fix** — a hardware-level reboot is outside this session's scope (no
  sudo/reboot action was taken; the machine was left in its current state
  for a user-driven reboot).

**Validity assessment of the office_1 capture above:** the 6 samples show
internally coherent, monotonically-improving behaviour (growing FREE count,
growing reachable pocket, real frontier acceptances, real pose displacement)
— this is not the signature a GPU-starved Unity render loop would produce
(the earlier xhost-bug run showed exactly that failure mode: frozen pose,
frozen grid, for comparison). Treated as **likely valid** but not
GPU-timestamp-confirmed for its full duration, since no continuous GPU
sampling ran alongside the debug dump. `core.heads.explore_debug` does not
currently record GPU state; if live captures continue to be a going concern,
consider adding a cheap `nvidia-smi`-derived clock/pstate field to the
per-record dump so future runs are self-certifying on this axis.

### Screenshot evidence — not obtained, and a privacy note

Attempted to save an RVIZ screenshot as visual evidence per a mid-task ask.
`xwd` failed (`BadColor` on GL-backed windows, both `-root` and by window id).
Installed `mss` into the host venv and captured a full-desktop PNG as a
fallback — **that screenshot showed an unrelated foreground application
window (an unrelated chat/coding session, not this task) occluding almost
all of RVIZ, with private conversational content from that other session
visible.** Deleted the file immediately (never written anywhere outside the
local scratch/report path, never committed, not otherwise distributed) and
did not retry desktop capture — this display is evidently shared with
another active session and full-screen capture on it is not safe. No
screenshot evidence exists in this repo. The numeric costmap/frontier
evidence in the table above (and the JSONL captures) is the substitute for
the "quantify what the map contains" ask; scene-extent-vs-object-list
comparison: office_1 objects span x in [-4.3, 6.1] / y in [-5.1, 1.7]
(~70 m^2 bbox), livingroom_1 objects span x in [-2.7, 2.3] / y in [-11.9, 2.6]
(~72 m^2 bbox but a much more elongated/corridor-like footprint) — from
`data/unity_scenes_ros2/<scene>/<scene>/object_list.txt`, not from a live
terrain capture (no livingroom_1 live run happened).

### Cleanup performed

- `kill -9` on every host adapter process (`ros_adapter.adapter_node` —
  including the leftover one the earlier `&&`-chained `pkill` missed) and
  the `ros2-daemon`; confirmed via `nvidia-smi --query-compute-apps` that no
  process still holds GPU memory.
- Inside `iros2026_system`: killed Unity/RVIZ/launch processes, then
  `docker stop iros2026_system`. `docker ps -a` confirms both
  `iros2026_system` and `iros2026_ai_module` are Exited.
- No Ollama process was started this session (host env has the LLM slot
  commented out; the adapter ran regex-only parse, per its own startup log).
- GPU state at end of session: reported idle (`P8, 0% util, ~4W`) but a
  compute probe still shows ~9x degraded throughput vs healthy — **left
  as-is for a user-driven reboot**, not chased further (no sudo/reboot
  action taken).

## Status / next steps

- **Phase 1 (instrumentation): done, committed.**
- **Phase 2 (comparison): half-done, and the half that matters most is the
  missing half.** office_1/inst captured and looks healthy — does not
  reproduce the starvation symptom (consistent with its 0.50 baseline
  score). livingroom_1/inst (the scene that actually exhibits the bug, 0.00
  baseline) was attempted but the capture is invalid — tainted by a
  confirmed, sustained GPU wedge from roughly 30s into the explore_execute
  phase onward. **The money comparison the issue asks for is still open.**
  Next session: reboot the machine first, then re-run the exact same
  procedure (scene swap, `xhost +`, sim up, host node with
  `VLA_EXPLORE_DEBUG_DIR` set, publish the livingroom_1/inst question with
  correct QoS, capture 120-180s) with continuous GPU clock/pstate logging
  alongside the debug dump (a cheap addition — see the note in the
  "Validity assessment" paragraph above) so any future wedge is
  self-documenting in the data instead of requiring a separate manual
  check.
- **Phase 3 (fix): not started.** No root cause identified — the one scene
  successfully captured (office_1) is the one that already works, so there
  is nothing to diagnose yet from live data. The instrumentation is in
  place and proven to produce useful signal (see the office_1 table); once
  a valid livingroom_1 capture exists, the same read (`sweeping`/`frontier`
  status transitions, frontier accept/reject reasons, FREE/pocket growth
  curve) should show WHERE it diverges from the office_1 pattern above —
  e.g. status stuck on `sweeping` past ~75s and never flipping to
  `frontier`, or flipping but every candidate rejected as
  `unreachable`/`score_below_min`, or the FREE count never growing past its
  first-sweep value. This is a concrete, falsifiable prediction for the
  next session to check, not a guess to act on now.
- No fix landed — nothing to re-verify against the 0.00 livingroom baseline
  yet.

## Reusable scripts (not committed — scratchpad only)

- `scene_swap.sh <scene>` — stop/start container, `docker cp` scene files,
  chown/chmod, restart. Does NOT run `xhost +` (see gotcha above — do that
  manually first).
- `live_question.sh "<question>" [duration_s]` — publishes
  `/challenge_question` with the QoS flags the adapter's subscription
  actually needs.

Both live in the session scratchpad, not this repo — promoting them into
`tools/` (with the `xhost +` step made explicit) would be a reasonable
follow-up so the next live session doesn't re-discover the same two gotchas.
