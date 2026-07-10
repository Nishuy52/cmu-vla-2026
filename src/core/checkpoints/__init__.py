"""LLM/VLM checkpoint protocol modules (architecture §3, design doc checkpoint_design.md).

Five bounded, timeout-guarded, ledger-capped checkpoints with structured JSON contracts
and deterministic fallbacks:

* CP2 :mod:`core.checkpoints.miss_recovery`   — detector-miss recovery (vision)
* CP3 :mod:`core.checkpoints.anchor_confirm`  — anchor confirmation on arrival (vision)
* CP4 :mod:`core.checkpoints.verification`    — pre-answer verification (text; highest value)
* CP5 :mod:`core.checkpoints.frontier_select` — frontier selection (vision, rare)

Each exposes a ``build_*(chat_fns, ledger, clock, cfg) -> callable`` matching the head
seam it is injected into. Shared JSON validation + one-repair decode lives in
:mod:`core.checkpoints.schemas`.
"""
