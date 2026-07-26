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

## Cluster queue — STATE 26 Jul 23:45
IN FLIGHT (do not cancel — cancelling loses the ~12h queue place):
- **698999** single livingroom_1 inst — validates the bag-hang fix + openai-installed
  LLM tier. est start 27 Jul 11:48. Harvest: `harvest_verify.sh 698999`.
- **699009** BATCH, 10 questions (livingroom_1 + office_1 x nume/obje/inst),
  chained `afterany:698999`, -t 2:55. Harvest: `harvest_verify.sh 699009`
  (auto-detects batch, scores every per-question bag).
Poller b13iwm5fu watches both; exits if the tunnel drops (harmless — outputs
persist on the cluster; just re-auth `ssh xlogin true` and harvest).

KEY OPEN QUESTION: does an LLM tier finally serve the parse? Check
`verify_run_698999_usage.jsonl` / `verify_batch_699009_usage_*.jsonl` and the
`parsed:tier=` line. SoCLaaS is unreachable from compute nodes (probe HTTP 000),
so expect ollama-local to serve; if tier=regex persists, that is still the blocker.

