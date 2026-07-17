# T12 Implementation Plan — true numerical yardstick, provenance, pre-Ubuntu queue

> **For agentic workers:** execute task-by-task via role subagents
> (executor / mech-executor / verifier per repo orchestration policy).
> Steps use checkbox (`- [ ]`) syntax for tracking. Each task's agent
> writes its deliverable to disk incrementally (T11 restart-proofing
> rule) and NEVER runs `git stash` / `git checkout` / `git reset`.

**Goal:** battery numbers become ground-truthed (true accuracy k/15),
born-provenanced, and diffable by tool; the 4 true numerical failures
get root-cause verdicts (+fixes where safe); pre-Ubuntu queue items
meth-F5 / arch-F9 / meth-F11 land.

**Architecture:** all battery-side changes concentrate in
`src/core/runner/` (one new `provenance.py` module shared by
`gt_battery.py` and `cvsweep.py`); the diff tool is an offline
`tools/` citizen outside the scored surface; diagnosis touches only
`src/core/groundtruth/` + resolver code it implicates, gated by the
11 currently-true answers as a regression floor.

**Tech stack:** existing pure-Python core; `subprocess` git calls only
inside runner/tools (never in scored `core/` paths that ship);
pytest fast tier while iterating, full gate before milestone commits.

**Branch:** `feat/t12-numerical-yardstick`. Spec: `task.md` beside
this file. Baseline for all before/after comparisons:
`reports/gt_battery_postT11_2026-07-14/gt_battery_results.json`.

Known repo facts agents need (verified 2026-07-17):

- Battery: `src/core/runner/gt_battery.py`. `GTQuestionScore` at
  :467; numerical scoring via `S.score_numerical` at :535;
  `aggregate()` at :746; `write_report()` payload at :969; CLI
  `main()` at :992 (`--no-spawn-hint` already exists at :1012).
- Run command (from `src/`):
  `python -m core.runner.gt_battery --groundtruth ..\data\vla3d\Unity --out ..\reports\<dir>`
- Calibration snapshot primitives: `src/core/calibration.py` —
  `default_calibration()`, `to_json()`, `diff()`.
- Sweep results payload: `src/core/runner/cvsweep.py:973`
  (`results.json` writer at :890).
- Answer key: `docs/gt_answers_numerical.json` — `scenes.<scene>`
  → `{question_raw, answer}`; `question_raw` has its whitespace
  stripped by PDF extraction (compare whitespace-insensitively).
  Exactly one numerical question per scene, 15 scenes.
- Battery tests: `src/tests/runner/test_gt_battery.py` (loft smoke),
  `test_battery.py`, `test_cvsweep.py`. Tools tests run separately:
  `python -m pytest tools` from repo root.
- 4 true failures (per postT11 run + answer key): arabic_room
  (sofas-below-window, GT 2, ours 0), loft (black-pillows-on-sofa,
  GT 2, undercount), home_building_1 (pillows-on-sofa-under-pictures,
  GT 6, ours 11), home_building_2 (red-pillows-on-sofa, GT 2, ours 3).
  Re-derive exact current numbers from the Task-4 run before fixing.

---

### Task 1 — provenance stamps + GT yardstick + topline honesty (executor)

**Files:**
- Create: `src/core/runner/provenance.py`
- Modify: `src/core/runner/gt_battery.py`, `src/core/runner/cvsweep.py`
- Test: create `src/tests/runner/test_provenance.py`; extend
  `src/tests/runner/test_gt_battery.py`

Two commits, in this order (meth-F7/F8 must exist before any new
battery generation).

**Part A — provenance (meth-F7/F8).**

- [ ] `provenance.py` exposes exactly:

```python
def collect_provenance(
    tool: str,
    argv: list[str] | None = None,
    cal: "Calibration | None" = None,
) -> dict:
    """Provenance stamp for results JSON (meth-F7/F8).

    Never raises: any git/subprocess failure degrades the affected
    fields to None and appends a human-readable reason to "note".
    """
```

  Returned dict schema (all keys always present):

```json
{
  "tool": "gt_battery",
  "generated_utc": "2026-07-17T04:12:00Z",
  "argv": ["--groundtruth", "..."],
  "git_commit": "<full hex or null>",
  "git_branch": "<name or null>",
  "git_dirty": true,
  "dirty_digest": "<sha1[:12] of porcelain-status + diff text; null when clean>",
  "calibration_sha1": "<sha1[:12] of core.calibration.to_json(cal)>",
  "calibration": { "<flattened calibration dict, the to_json payload>" : "..." },
  "note": ""
}
```

  Git via `subprocess.run(["git", ...], cwd=<repo root>)` with repo
  root derived from `Path(__file__).resolve().parents[3]`. `cal=None`
  means `default_calibration()`.
- [ ] `gt_battery.write_report()` payload gains
  `"provenance": collect_provenance("gt_battery", argv, cal)` — thread
  `argv` from `main()` (new keyword param, default `None`, so library
  callers/tests need no change).
- [ ] `cvsweep` `results.json` payload (:973) gains the same stamp
  with `tool="cvsweep"` plus the swept/frozen config identity it
  already tracks.
- [ ] Per-question IF legs into results (meth-F7 second half):
  `GTQuestionScore` gains `leg_goals: list | None = None`
  (`[[kind, [x, y]], ...]`, from `_if_rubric_geometry`) and
  `leg_outcomes: list | None = None` — one dict per leg:
  `{"i": 0, "kind": "goto", "goal": [x, y], "reached_in_order": bool,
  "threaded": bool | null}`. Read the rubric scorer in
  `src/core/groundtruth/scoring.py` to expose per-leg results; if it
  only returns totals today, extend its return object — do NOT
  recompute legs in the battery.
- [ ] Tests (`test_provenance.py`): stamp has every schema key; runs
  green inside this repo (git_commit non-null, calibration_sha1 is
  12 lowercase hex); graceful-degradation path (monkeypatch
  `subprocess.run` to raise → keys present, nulls + note, no raise).
  Extend the loft smoke test: results JSON has `provenance.tool ==
  "gt_battery"` and every numerical row carries the Part-B fields.
- [ ] Run: `python -m pytest tests/runner -q` from `src/` → green.
- [ ] Commit: `battery/cvsweep: provenance stamps + per-leg IF outcomes in results (meth-F7/F8)`

**Part B — GT yardstick + topline honesty (arch-F3 wiring, meth-F4/F6).**

- [ ] New CLI arg `--answers`, default
  `<repo>/docs/gt_answers_numerical.json` (module constant
  `DEFAULT_ANSWERS` beside `DEFAULT_QUESTIONS`). Missing file →
  battery still runs; true-fields null; topline says
  `true accuracy n/a (no answer key)`.
- [ ] `GTQuestionScore` numerical rows gain `gt_answer_true:
  int | None`, `true_match: bool | None`, `true_source: str`
  (`"questions_pdf_text"` on success, else `""`). Matching rule:
  by scene (one numerical question each), then GUARD by comparing
  whitespace-stripped casefolded `question_raw` vs the battery
  question; mismatch → leave fields null, note
  `answer-key question mismatch` (never silently mis-anchor).
- [ ] `aggregate()["numerical"]` gains `n_with_true_answer` and
  `true_accuracy` (mean of `true_match` over rows with a true
  answer) and renames `exact_match_rate_pipeline` →
  `pipeline_determinism_rate` (meth-F4: determinism, not accuracy).
  Per-row JSON field `exact_match` keeps its name.
- [ ] Topline edits in `write_report()` (meth-F4+F6): numerical line
  LEADS with `TRUE accuracy k/15 (answer key: questions.pdf)`;
  pipeline determinism dropped from the topline sentence (JSON keeps
  it); OR line becomes `instance-match k/8 scored (scoreability
  8/30); IoU pending real perception` — instance-match = resolved
  instance id equals `gt_target_id` (add `our_target_id` to the row
  if `score_object_reference` exposes it; if it does not, extend it);
  `main()`'s stdout summary swaps `num_exact=` for `num_true=`.
- [ ] Tests: answer-key loading (loft smoke: `gt_answer_true == 2`,
  `true_source == "questions_pdf_text"`); mismatch guard (tampered
  key text → null + note); aggregate math on synthetic scores
  (2 true + 1 false + 1 keyless → `true_accuracy == 0.6667`,
  `n_with_true_answer == 3`).
- [ ] Run: `python -m pytest tests/runner -q` from `src/`, then fast
  tier `python -m pytest` from `src/` → green.
- [ ] Commit: `battery: true numerical yardstick from answer key + topline honesty (arch-F3, meth-F4/F6)`

### Task 2 — battery_diff tool (mech-executor, parallel with Task 1)

**Files:**
- Create: `tools/battery_diff.py`
- Test: create `tools/test_battery_diff.py` (follow existing tools/
  test conventions; suite runs via `python -m pytest tools` from repo
  root, NOT collected by `src/`)

- [ ] CLI: `python -m tools.battery_diff <before_results.json>
  <after_results.json> [--out <path.md>]` (default: stdout). Inputs
  are `gt_battery_results.json` files.
- [ ] Output (markdown, and the ONLY legal source of before/after
  tables in reports, per meth-F8):
  1. Provenance header for BOTH inputs — tool, generated, commit,
     dirty, calibration_sha1; a file with no `provenance` key prints
     `UNPROVENANCED (pre-meth-F7 run)`.
  2. Topline table: every numeric leaf under `aggregate.*`
     (dotted keys), columns Before / After / Δ; keys present on only
     one side shown with `—` on the other (tolerates schema drift,
     e.g. postT11 lacking `true_accuracy`).
  3. Changed-rows table: rows matched by `(scene, qtype, question)`;
     a row is changed iff any scored field differs; show the changed
     fields only, `old -> new`.
  4. Added/removed rows listed by key.
- [ ] Pure stdlib; no imports from `src/` (tools are outside the
  challenge-fork surface and must not couple to core).
- [ ] Tests with two small synthetic results dicts written to tmp:
  detects a changed `our_count`, an added row, a topline delta, and
  the UNPROVENANCED label; `--out` writes the file; identical inputs
  → "no changes" and exit 0.
- [ ] Run: `python -m pytest tools -q` from repo root → green.
- [ ] Commit: `tools: battery_diff — sole legal source of before/after battery tables (meth-F8)`

### Task 3 — IF leg-count census fixture (executor, parallel)

**Files:**
- Create: `src/tests/groundtruth/fixtures/if_leg_census.json`
- Test: create `src/tests/groundtruth/test_leg_census.py`
  (if `src/tests/groundtruth/` does not exist, place both under the
  existing directory that holds the parsing/groundtruth tests —
  follow the tree, do not invent a new top-level)

- [ ] Census (meth-F5): for each of the 30 IF questions in
  `upstream/CMU-VLN-Challenge-2026/questions/questions.json`
  (2 per scene), read the TEXT and hand-tally: `n_route_legs`
  (ordered goto/via/corridor clauses) and `n_avoid` (avoid clauses).
  Record `{"scene": ..., "q_index": 0|1, "question": ...,
  "n_route_legs": k, "n_avoid": m, "clauses": ["...", ...]}` — the
  `clauses` list is the human-auditable tally, one string per
  counted clause. This is a REFERENCE fixture: derive it from
  reading the question text, never from running the parser.
- [ ] Test: for every fixture row, run the pipeline's IF parse
  (the same path the battery uses — see `_if_rubric_geometry` and
  the plan builder it calls) and assert
  `len(plan.route) == n_route_legs` and
  `len(plan.avoid) == n_avoid`. Every currently-failing row gets
  `pytest.param(..., marks=pytest.mark.xfail(strict=True,
  reason="dropped-leg: <scene> q<idx>"))` so the census documents
  today's gaps without reddening the gate — and any silent
  improvement/regression flips loudly (strict).
- [ ] Skip cleanly when `upstream/` is absent (existing pattern in
  `src/tests/runner/` — reuse it).
- [ ] Run: the new test file + fast tier from `src/` → green.
- [ ] Commit: `tests: IF leg-count census fixture vs plan.route/plan.avoid (meth-F5)`

### Gate A — verifier over Tasks 1–3

- [ ] Dispatch verifier: claims = (1) provenance stamp present +
  schema-complete in a freshly generated loft-scene battery results
  file, degradation path safe; (2) true-yardstick fields + topline
  match the answer key on loft; renames didn't break cvsweep or
  existing tests; (3) battery_diff output contains the four sections
  on synthetic input and labels postT11 UNPROVENANCED against a
  fresh run; (4) census fixture rows spot-check against question
  text (verifier re-tallies 5 random rows independently); xfail
  marks are strict. Full gate `python -m pytest -m "" -n auto` from
  `src/` + `python -m pytest tools` from repo root.
- [ ] REFUTED findings → back to the owning executor (max 2 rounds,
  then orchestrator takes over).

### Task 4 — provenanced baseline battery run (orchestrator-run, background)

- [ ] From `src/`:
  `python -m core.runner.gt_battery --groundtruth ..\data\vla3d\Unity --out ..\reports\gt_battery_T12pre_2026-07-17`
- [ ] Confirm: results JSON carries provenance (commit = Task-1/2/3
  merge state, dirty=false); `true_accuracy` reproduces 11/15
  (0.7333); the 4 false rows are exactly the four known scenes.
  If not exactly those four → STOP, update task.md Context with the
  actual set before Task 5 dispatches (the worklist follows the
  data, not the prose).
- [ ] `python -m tools.battery_diff reports/gt_battery_postT11_2026-07-14/gt_battery_results.json reports/gt_battery_T12pre_2026-07-17/gt_battery_results.json`
  → expect: no per-row scoring changes (instrument unchanged), new
  aggregate keys appear one-sided, postT11 labeled UNPROVENANCED.
- [ ] Commit the report dir: `reports: T12pre provenanced baseline battery (true accuracy 11/15)`

### Task 5 — diagnose + fix the 4 true failures (executor; the core judgment task)

**Files:**
- Create: `docs/tasks/T12-numerical-yardstick/diagnosis.md`
- Modify: whatever the diagnosis implicates — expected candidates are
  `src/core/groundtruth/scoring.py`, the numerical resolver /
  relation predicates under `src/core/` (T7's cause-bucket record at
  `docs/tasks/T7-numerical-count-diagnosis/` maps this terrain —
  read it first), plus tests beside the touched code

- [ ] For each failure (arabic_room 0-vs-2 sofas-below-window;
  loft black-pillows undercount; home_building_1 11-vs-6
  pillows-on-sofa-under-pictures; home_building_2 3-vs-2
  red-pillows-on-sofa): reproduce in isolation
  (`--scenes <scene>`), trace the resolver decision (candidate set →
  relation filter → color/vocab filter → count), and write a verdict
  in `diagnosis.md` using T7's buckets (resolver-code defect /
  calibration / GT-index artifact / question-semantics), with the
  exact instance ids kept and dropped.
- [ ] Fix ONLY what has a clear code-defect or semantics verdict.
  HARD REGRESSION GATE: after each fix, rerun the battery numerical
  slice over ALL 15 scenes
  (`python -m core.runner.gt_battery --groundtruth ..\data\vla3d\Unity --no-drive-if --out ..\reports\scratch_t12_num` from `src/`;
  scratch dirs stay uncommitted) — the 11 true passes must all hold;
  a fix that trades a pass for a pass is a net loss, revert it and
  record the verdict as diagnosed-unfixed. No calibration-value
  tuning to chase single questions (that is the sweep's job, and it
  is deferred to real sim).
- [ ] Every fix lands with a unit test pinning the corrected
  behavior at the resolver level (not only via the battery).
- [ ] Run fast tier from `src/` green; write `diagnosis.md`
  incrementally (restart-proofing).
- [ ] Commit per fix + final: `numerical: <scene> root-cause fixes — true accuracy k/15 (T12 diagnosis)`

### Task 6 — --no-spawn-hint battery run (mech-executor, parallel with Task 5)

- [ ] From `src/`:
  `python -m core.runner.gt_battery --groundtruth ..\data\vla3d\Unity --no-spawn-hint --out ..\reports\gt_battery_nospawnhint_2026-07-17`
  (arch-F9: bound exploration sensitivity; first committed use of
  the knob). Run AFTER Gate A so it is born provenanced; it reads
  code only, so it can run concurrently with Task 5's scratch runs
  as long as output dirs differ.
- [ ] Append a short section to the run's `gt_battery_report.md`? No —
  reports are generated files. Instead: one paragraph in
  `docs/tasks/T12-numerical-yardstick/task.md` Notes comparing IF
  topline vs the T12pre run **using battery_diff output only**
  (`python -m tools.battery_diff reports/gt_battery_T12pre_2026-07-17/gt_battery_results.json reports/gt_battery_nospawnhint_2026-07-17/gt_battery_results.json`),
  pasted as the machine table + at most 3 sentences of reading below
  it (meth-F8 discipline: prose interprets below the table, never
  introduces numbers above it).
- [ ] Commit: `reports: --no-spawn-hint battery run — exploration-sensitivity bound (arch-F9)`

### Task 7 — 3 unaligned IF scenes: frame fit or confirmed exclusion (executor)

**Files:**
- Create: `docs/tasks/T12-numerical-yardstick/unaligned_scenes.md`
- Possibly modify: the frame-fit path in `gt_battery.py`
  (`score_scene`'s scene-level sim→object transform fit, :561–:585
  region) and/or a per-scene correspondence fixture

- [ ] Identify the 3 scenes from `aggregate.instruction_following
  .unaligned_scenes` in the T12pre results (do not trust prose).
- [ ] For each: why does the endpoint fit fail (`fit_residual_m`,
  endpoint correspondence quality, trajectory direction)? Try
  alternative fits: more correspondence points from the GT
  trajectory PLY, RANSAC over endpoint pairs, per-question instead
  of per-scene fit. This is meth-F11's "one-time manual fit" — a
  committed per-scene transform fixture is an acceptable outcome if
  it is derived reproducibly (script or documented procedure in
  `unaligned_scenes.md`).
- [ ] Outcome A (rejoin): scenes score with `frame_aligned=true`,
  residual sane (< 1 m), IF aggregate now covers them — document the
  before/after via battery_diff. Outcome B (exclusion confirmed):
  written evidence that the GT trajectories cannot be fit (bad data,
  not our resolution) — the friendly-ward bias hole is then closed
  as DATA, and the report's unaligned note should say so.
- [ ] Fast tier green; commit:
  `battery: unaligned-scene frame fit — <rejoined|exclusion confirmed> (meth-F11)`

### Gate B — verifier over Tasks 5–7

- [ ] Dispatch verifier: (1) re-run the final battery numerical
  slice, confirm claimed true accuracy and zero regressions among
  the original 11; (2) audit each diagnosis verdict against the code
  it blames (spot-check 2 of 4 deeply); (3) confirm the no-spawn-hint
  paragraph's numbers all appear in its battery_diff table; (4)
  re-derive the unaligned-scene verdict for 1 of the 3 scenes.
  Full gate from `src/` + tools suite.

### Task 8 — final battery + close-out (orchestrator + mech-executor for docs)

- [ ] Final full run:
  `python -m core.runner.gt_battery --groundtruth ..\data\vla3d\Unity --out ..\reports\gt_battery_T12post_2026-07-17`
  (from `src/`); commit the dir.
- [ ] `python -m tools.battery_diff reports/gt_battery_postT11_2026-07-14/gt_battery_results.json reports/gt_battery_T12post_2026-07-17/gt_battery_results.json --out docs/tasks/T12-numerical-yardstick/postT11_to_T12.md`
  — the born-provenanced postT11→T12 comparison the adjudication
  asked for; commit.
- [ ] Docs (mech-executor): tick T12 acceptance boxes in `task.md` +
  dated closing note; add T12 row to `docs/tasks/INDEX.md`; update
  `docs/redteam/hardening_backlog.md` ledger rows (meth-F4, F5, F6,
  F7, F8, F11, arch-F9, arch-F3-numerical → landed/partial with
  pointers); LOG.md one-liner (task-milestone format, pointer to the
  task folder); `docs/ubuntu_setup.md` ONLY if dependencies changed
  (none expected — battery_diff is stdlib).
- [ ] Full gate: `python -m pytest -m "" -n auto` from `src/` AND
  `python -m pytest tools` from repo root → green.
- [ ] Commit, push `origin feat/t12-numerical-yardstick`, open PR to
  `main` (no AI/tooling attribution anywhere, including the PR body).

---

## Self-review (done at write time)

- Spec coverage: scope items 1→Task 1B, 2→Tasks 1A+2, 3→Tasks 4+5,
  4→Tasks 3+6+7; acceptance criteria map to Gates A/B + Task 8. ✓
- No placeholders: every task names exact files, commands, schemas;
  judgment tasks (5, 7) specify method + verdict format instead of
  code, per repo policy that executors own local design. ✓
- Type consistency: `collect_provenance` signature matches both call
  sites; `gt_answer_true`/`true_match`/`true_source` names identical
  across Tasks 1B, 4, 5; diff keys `(scene, qtype, question)` match
  the results schema. ✓
