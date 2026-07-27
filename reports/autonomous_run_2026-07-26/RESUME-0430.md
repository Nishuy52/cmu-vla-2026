# Cold-pickup — resume 28 Jul 2026 ~04:30 SGT

Written 28 Jul ~01:45 before a session limit. Everything below is committed and pushed;
nothing is lost if this session dies. `main` == `origin/main`, working tree clean.

## 1. Where we actually are (the honest number)

45 live-scored questions, 15 scenes, all 3 question types. All ran clean
(45/45 SUCCESS, every parse `tier=api` via SoCLaaS).

| type | live | offline | detail |
|---|---|---|---|
| instruction_following | **0.367** | 0.800 | 4 perfect, 8 zero (15 scenes) |
| numerical | **0.133** | 1.000 | 2/15 correct |
| object_reference | **0.005** | 1.000 | 7/15 scoreable, 0 perfect |
| all 45 | **0.167** | — | 37/45 scoreable |

**The bottleneck is perception + grounding.** The FSM, exploration, LLM tier, bag
capture and scoring harness all work now. With GT-perfect perception we score
0.80/1.00/1.00; with the live stack, 0.37/0.13/0.005.

**Today's fixes did NOT move the live IF score**: pre-fix 699817 = 0.375 (8 scenes),
post-fix 699818 = 0.357 (7 scenes). Different scene sets so not a clean A/B, but
no visible gain from #107.

## 2. In flight when the session ended

**Diagnosis workflow** `wf_d5c85523-ce5` (run id), 4 dimensions + adversarial
verification + synthesis:
- `obje-split` — per question: wrong-instance (grounding) vs wrong-extent (fusion)
  vs wrong-class (detection) vs implausible-marker (frame-fit). **This aggregate is
  the deliverable that decides where effort goes.**
- `numerical` — per-scene overcount/undercount classification; explicit verdict on
  whether #93 and #94 fired
- `instruction-following` — did #107 work (use `office_1`, present in BOTH pre- and
  post-fix runs, as the sound same-scene comparison); what distinguishes the 4
  perfect scenes
- `plumbing` — why #98/#102 dumps captured nothing

Transcript: `~/.claude/projects/-home-jason-cmu-ws/<session>/subagents/workflows/wf_d5c85523-ce5/`
If it completed, its result is in that dir's `journal.jsonl`. If it did not, re-run:
`Workflow({scriptPath: ".../workflows/scripts/live-failure-diagnosis-wf_d5c85523-ce5.js", resumeFromRunId: "wf_d5c85523-ce5"})`
— completed agents return cached results instantly.

## 3. Do this at 04:30, in order

1. **Collect the diagnosis** (above). Then **file one issue per confirmed finding**
   (standing rule: every defect AND every proposed fix gets its own verified issue —
   see the FINDINGS-INDEX pattern in `reports/cluster_verify/699009/`).
2. **Act on the obje A/B/C/D split** — it tells you whether to spend effort on
   grounding (#91/#107) or fusion (#94). object_reference is 30 of 75 questions and
   currently scores ~0.
3. **Fix #119 before any new cluster submission**: derive new dump paths from the
   existing `VLA_EXPLORE_DEBUG_DIR` rather than new env vars, so #98/#102 actually
   capture next time. A queued job CANNOT pick up a new env var (frozen sbatch).
4. **Next sweep** — only after a fable pre-submit verification (standing rule).
   Cluster is idle, all 15 scenes uploaded, queue empty.

## 4. Hard-won facts — do not rediscover these

- **Never `scancel` to apply a fix** — loses the ~12h queue place. The sbatch body is
  frozen at submit; only runtime-loaded files update (`~/vla/src`, `~/live_monitor.py`,
  `~/question_pub.py`, `~/topic_probe.py`). Use `scontrol hold`/`release` to close the
  race when syncing `src` between jobs, and `scontrol update` for time limits.
- **Always fable-verify batch scripts before submitting.** This caught two silent
  run-invalidators (unexpanded `~` meaning the scene overlay never happened; one
  shared matrix filename meaning all chained jobs would read the last matrix).
- **Bags never finalize on SIGINT** (#114) — every capture needs
  `ros2 bag reindex -s mcap` + `ros2 bag convert`. Fixed in-job for FUTURE
  submissions only; 699817-699820 needed manual recovery.
- **Schema trap**: in `instance_index.jsonl`, `by_class`/`total_instances` are
  TOP-LEVEL, not under `live_instances`. A wrong path silently returns empty.
- **`score_live_run` without `--out` overwrites the committed 20-Jul baseline** (#96).
  Always pass `--out`.
- **`_merge_with_existing` keys by (scene, qdir)** (#97) — scoring 2 questions of the
  same scene+type keeps only the last row. Console output has the full set.
- SoCLaaS is validated live: 100% success, ~2.6k tokens/question, latency ~7s
  (max 13.2s). Usage accumulates in `reports/soclaas_usage.jsonl`.

## 5. Data locations

- `reports/cluster_verify/699817/` 8 inst (PRE-fix) · `699818/` 7 inst (POST-fix)
- `reports/cluster_verify/699819/` 15 nume · `699820/` 15 obje (both POST-fix)
- Each: `captures/<slot>/<scene>/<qdir>/bag` (recovered, readable), `captures/scores.json`,
  `debug/<slot>/{instance_index,raw_detections,explore_debug_*}.jsonl` — instance_index
  now carries `aabb_min`/`aabb_max` (#101).
- `reports/gt_battery_postfix_2026-07-27/` — current offline reference
  (obje scoreable 6/30 → **12/30**, IF 0.744, numerical 15/15).
- Earlier analyses: `reports/cluster_verify/699009/ANALYSIS-*.md` + `FINDINGS-INDEX.md`.
- mcap files are gitignored (local only, ~5GB).

## 6. Open issues: 21

Filed today from live evidence: #91 (missing-class disambiguators — dominant IF cost),
#92 (obje scorer resolution), #93 (relation clause not restricting), #94
(under-segmentation + z error), #95 (disambiguator recursion), #96/#97 (score_live_run
data loss), #98/#101/#102 (observability), #100 (PASS-BY tolerance), #103 (nav outside
explored region), #104 (marker outside GT extent), #114 (bag finalize), #118 (obje
answers wrong, not just unscoreable), #119 (dumps captured nothing).
