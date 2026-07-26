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

## Cluster queue (serial)
1. [PENDING in SLURM queue] job 697761 — livingroom_1 inst — bag-path validation + #82/#84/#83 + first real IF-rubric score (poller b7rgt25w5)
2. [ ] office_1 inst
3. [ ] livingroom_1 nume (#89 counts + numerical-match)
4. [ ] office_1 nume
5. [ ] livingroom_1 obje (IoU)
6. [ ] office_1 obje
7. [ ] arabic_room upload + inst (#85)

## Host lanes (parallel executor worktrees)
- A [dispatched] SoCLaaS default primary + usage logging (core/llm, adapter, docs) -> reports/soclaas_usage.jsonl
- B [dispatched] #90 float32 lattice collision (core/nav/occupancy.py)
- C [dispatched] #81 offline battery budget hooks so head gates fire (core/runner/gt_battery.py)
- D [dispatched] #84 raw-detection dump + #89 subphrase folding (core/perception)

## Harvest/integrate checklist
- Each worktree agent: collect final message, integrate to main, commit/push, delete worktree.
- Each cluster job: harvest_verify.sh <job> -> score -> log numbers -> comment issue.
