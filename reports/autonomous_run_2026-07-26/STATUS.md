# Autonomous run — 2026-07-26 (session 23 continuation)

Cold-pickup state for the 12h autonomous plan (user-approved). Update at every milestone.

## Approved scope
- SoCLaaS = **permanent primary LLM tier** (approved architectural change) + usage monitoring accumulating across sessions.
- Phase-2 host fixes: #90, #81, #84 instrumentation, #89.
- Maximize cluster live scoring runs while the tunnel is up (serial — shared udocker unity dir).

## Guardrails
- Cluster jobs SERIAL (scene overlay writes shared ~/.udocker unity dir on NFS home).
- Close issues only on measured numbers (conservative; prefer "ready to close" recs).
- Commit/push per milestone; keep host lane going if tunnel drops.
- SSH ControlMaster re-auth is HUMAN-ONLY — if it expires, cluster lane pauses, host lane continues.

## Cluster queue (serial) — STATE 26 Jul 16:10
FIRST LIVE SCORE OBTAINED: 697761 livingroom_1 inst IF-rubric = **0.000** (offline
0.744), legs 0/2, frechet 4.26m, coverage 0. #83 cured starvation (FSM explores/
answers/drives full route) but the route scores 0 — wrong grounding/routing. Parse
fell to tier=regex (SoCLaaS+ollama both configured, neither served; 697761 = old
src, no usage log). Bag SLURM-killed@25min, recovered via `ros2 bag reindex` +
`ros2 bag convert`. Score at reports/cluster_verify/697761/captures/scores.md.

ACTIVE: job **698278** livingroom_1 inst — PENDING, backfill est 2026-07-27T18:18
(~26h, could backfill earlier). Poller bl8jorhqp (safety net). Runs CURRENT main src
(all fixes) + fixed -t 40 + VLA_RAW_DETECTION_DUMP_PATH + VLA_LLM_USAGE_LOG +
compute-node SoCLaaS gateway probe. This is THE run that validates #84 raw dump /
#89 folding / SoCLaaS-serving / usage log / clean scoreable bag.
HARVEST when it runs (tunnel-independent — outputs persist on cluster ~/):
  source .venv/bin/activate && tools/cluster/live_run/harvest_verify.sh 698278
  (if bag truncated: ssh reindex+convert per 697761 procedure in LOG)
Check first: `SOCLAAS: compute-node gateway probe HTTP <code>` in the .out, and
verify_run_698278_usage.jsonl — tells us WHY parse fell to regex.

Remaining after 698278: office_1 inst, livingroom/office nume+obje, arabic (#85).
All serial (shared udocker unity dir). Queue is congested — each is a long wait.

## Cluster queue (serial)
1. [PENDING ~1h45m+, backfill est 18:48 — job 697761 livingroom_1 inst — sparse poller bcjlvzfqz]
2. [ ] office_1 inst
3. [ ] livingroom_1 nume (#89 counts + numerical-match)
4. [ ] office_1 nume
5. [ ] livingroom_1 obje (IoU)
6. [ ] office_1 obje
7. [ ] arabic_room inst (#85) — scene UPLOADED to cluster, ready

## Host lanes (parallel executor worktrees)
- A [DONE -> main c1efcc0] SoCLaaS default primary + usage logging (reports/soclaas_usage.jsonl)
- B [DONE -> main f6f913e, commented, worktree retired] #90 float32 lattice collision
- C [DONE -> main 97131d8; delta measuring in bg bcwvfg2wd] #81 budget hooks opt-in (default off = byte-preserving)
- D [DONE -> main 119aff9] #84 dump + #89 folding
- ALL HOST LANES DONE. #81 delta: gates-off 0.744 / gates-on 0.622 -> keep default off (commented).

## Verified ready
- Host scoring path OK: rosbags import, GT scenes, questions/answers, score_live_run all present in .venv.
- arabic_room uploaded to cluster; ollama qwen2.5vl:3b present; SoCLaaS gateway 200 + qwen3.6:35b valid.

## Harvest/integrate checklist
- Each worktree agent: collect final message, integrate to main, commit/push, delete worktree.
- Each cluster job: harvest_verify.sh <job> -> score -> log numbers -> comment issue.
