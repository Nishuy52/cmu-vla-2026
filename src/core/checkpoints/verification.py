"""CP4 — pre-answer verification (text). The SORT3D compositional-slip counter.

The highest-value checkpoint (design doc §CP4): guards the OR 2-pt and 6-pt types by
re-reading the *whole* question against the toolbox's computed per-clause facts, catching
constraints the deterministic clause list MISSED (the documented clause-slip failure
mode). Runs once (ledger cap 1), before committing the OR answer.

Contract (strict JSON, one repair, then deterministic fallback = keep the toolbox winner):

    {"verdict": "confirm"|"runner_up"|"neither",
     "missed_constraint": str|null,
     "reason": str}

Verdict handling (exactly per design):
* ``confirm``  -> keep the winner.
* ``runner_up``-> swap to the runner-up (once); if there is no runner-up, keep the winner.
* ``neither`` + ``missed_constraint`` -> if remaining time >= 90 s, RE-RESOLVE with the
  missed constraint appended as a text-matched clause and take that result's new winner;
  else keep the winner. ``neither`` without a missed_constraint keeps the winner.

Seam gap (documented, not patched here): the head's injected ``llm_verify`` seam is
``LlmVerifyFn = (plan, candidate_summary, pass_matrix_text) -> bool`` — narrower than CP4:
it carries no runner-up, no ``plan.notes``, and expects a keep/demote bool rather than the
three-way verdict + missed-constraint. :func:`build_verifier` therefore returns TWO
callables:

* ``build_verifier(...)`` (this module's primary API) returns the rich CP4 function
  ``run(question, winner, runner_up, pass_matrix, notes, remaining_s, resolve_again) ->
  VerificationOutcome`` for a future head that consumes the full contract, AND
* ``as_llm_verify_seam(...)`` adapts it to today's ``(plan, summary, matrix_text) -> bool``
  by degrading gracefully: it maps ``confirm`` -> keep (True) and ``runner_up``/``neither``
  -> demote (False) on the winner candidate, discarding the missed_constraint/re-resolve
  path the seam cannot express. The head then does its own runner-up walk. This keeps
  today's tests green while the full CP4 logic is unit-tested directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.checkpoints import schemas
from core.checkpoints._runtime import guarded_call

CHECKPOINT_NAME = "verification"

#: Design rule: re-resolve on a missed constraint only when this much time remains.
RE_RESOLVE_MIN_REMAINING_S: float = 90.0

SYSTEM_PROMPT = (
    "You verify a robot's selected answer object against every requirement in a spatial "
    "question. You are the last check before the answer is committed. Reply with STRICT "
    "JSON only."
)

PROMPT_TEMPLATE = """\
Question: {question}
Selected: {winner}
Computed facts per clause:
{fact_table}
Runner-up: {runner_up}
The parser noted: {notes}
Check EVERY requirement in the question against the computed facts, including any the \
clause list may have MISSED. Reply JSON only: \
{{"verdict": "confirm"|"runner_up"|"neither", "missed_constraint": str|null, "reason": str}}"""


@dataclass(frozen=True)
class VerificationOutcome:
    """Result of one CP4 run.

    action:   "keep" | "runner_up" | "re_resolve" — what the head should do.
    winner:   the object to answer with after applying the verdict.
    verdict:  the raw model verdict (or "fallback" when the call was skipped/failed).
    missed_constraint / reason: audit fields (may be None/"").
    """

    action: str
    winner: Any
    verdict: str
    missed_constraint: str | None = None
    reason: str = ""


# --------------------------------------------------------------------------- prompt build


def build_prompt(
    question: str,
    winner_summary: str,
    fact_table: str,
    runner_up_summary: str,
    notes: str,
) -> list[dict[str, str]]:
    """Assemble the CP4 fact-table messages from the resolve result + plan notes."""
    user = PROMPT_TEMPLATE.format(
        question=question or "(none)",
        winner=winner_summary or "(none)",
        fact_table=fact_table or "(no clauses computed)",
        runner_up=runner_up_summary or "(none)",
        notes=notes or "(none)",
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def fact_table_text(pass_matrix_rows: list) -> str:
    """Render a candidate's per-clause PredResult rows as `clause | passed | explanation`.

    Accepts the toolbox ``pass_matrix[id]`` list (each element exposing ``.passed`` and
    ``.explanation``). One row per clause, numbered, so the model can point at a slip.
    """
    if not pass_matrix_rows:
        return "(no clauses)"
    lines = []
    for i, r in enumerate(pass_matrix_rows):
        passed = "PASS" if getattr(r, "passed", False) else "FAIL"
        expl = getattr(r, "explanation", str(r))
        lines.append(f"  clause {i}: {passed} | {expl}")
    return "\n".join(lines)


def candidate_summary(cand: Any) -> str:
    """Compact one-line summary of a candidate instance for the prompt."""
    if cand is None:
        return "(none)"
    iid = getattr(cand, "instance_id", "?")
    label = getattr(cand, "label", "?")
    c = getattr(cand, "centroid", None)
    if c is not None:
        try:
            xyz = f"({float(c[0]):.2f},{float(c[1]):.2f},{float(c[2]):.2f})"
        except Exception:  # noqa: BLE001
            xyz = str(c)
    else:
        xyz = "?"
    return f"#{iid} ({label}) at {xyz}"


# --------------------------------------------------------------------------- core CP4


def run_verification(
    chat: Callable[[list[dict[str, str]]], str],
    ledger,
    clock,
    *,
    question: str,
    winner: Any,
    runner_up: Any,
    winner_facts: list,
    notes: str,
    remaining_s: float,
    resolve_again: Callable[[str], Any] | None = None,
    repair: Callable[[str, list], str] | None = None,
) -> VerificationOutcome:
    """Run one ledger-gated, timeout-bounded CP4 verification and apply the verdict.

    ``resolve_again(missed_constraint) -> new_winner`` is the head-supplied re-resolve hook
    (append the missed constraint as a text-matched clause, re-run resolve, return its new
    top candidate or None). ``remaining_s`` is the question-budget time left (the ledger's
    ``cfg`` callable feeds this in via :func:`build_verifier`).

    Deterministic fallback (skip/timeout/malformed-after-repair): keep the winner.
    """
    prompt = build_prompt(
        question,
        candidate_summary(winner),
        fact_table_text(winner_facts),
        candidate_summary(runner_up),
        notes,
    )
    raw = guarded_call(
        CHECKPOINT_NAME, ledger, clock, lambda: chat(prompt), tier="text"
    )
    if raw is None:  # ledger denied, timeout, or dead provider
        return VerificationOutcome("keep", winner, "fallback", None, "verification unavailable")

    obj, _errors = schemas.parse_with_repair(raw, schemas.validate_verification, repair)
    if obj is None:  # malformed even after one repair
        return VerificationOutcome("keep", winner, "fallback", None, "invalid verification reply")

    verdict = obj["verdict"]
    missed = obj.get("missed_constraint")
    reason = obj.get("reason", "")

    if verdict == "confirm":
        return VerificationOutcome("keep", winner, verdict, missed, reason)

    if verdict == "runner_up":
        if runner_up is not None:
            return VerificationOutcome("runner_up", runner_up, verdict, missed, reason)
        return VerificationOutcome("keep", winner, verdict, missed, reason)

    # verdict == "neither"
    if missed and remaining_s >= RE_RESOLVE_MIN_REMAINING_S and resolve_again is not None:
        try:
            new_winner = resolve_again(missed)
        except Exception:  # noqa: BLE001 — a failing re-resolve keeps the winner
            new_winner = None
        if new_winner is not None:
            return VerificationOutcome("re_resolve", new_winner, verdict, missed, reason)
    return VerificationOutcome("keep", winner, verdict, missed, reason)


# --------------------------------------------------------------------------- builders


def build_verifier(
    chat_fns,
    ledger,
    clock,
    cfg: dict | None = None,
) -> Callable[..., VerificationOutcome]:
    """Build the rich CP4 verifier bound to a ChatFn, ledger, clock, and config.

    ``chat_fns`` may be a bare text ChatFn or a mapping/object exposing ``chat`` (or
    ``text``); we resolve the text callable from it. ``cfg`` keys:

    * ``remaining_s``: a zero-arg callable returning the question-budget seconds left
      (per design, the re-resolve rule needs this). Defaults to +inf (always eligible).
    * ``repair``: optional (raw, errors) -> raw repair callable for the one repair round.

    Returns ``run(question, winner, runner_up, winner_facts, notes, resolve_again=None)``.
    """
    cfg = cfg or {}
    chat = _resolve_chat(chat_fns)
    remaining_fn: Callable[[], float] = cfg.get("remaining_s") or (lambda: float("inf"))
    repair = cfg.get("repair")

    def run(
        question: str,
        winner: Any,
        runner_up: Any = None,
        winner_facts: list | None = None,
        notes: str = "",
        resolve_again: Callable[[str], Any] | None = None,
    ) -> VerificationOutcome:
        return run_verification(
            chat,
            ledger,
            clock,
            question=question,
            winner=winner,
            runner_up=runner_up,
            winner_facts=winner_facts or [],
            notes=notes,
            remaining_s=_safe_remaining(remaining_fn),
            resolve_again=resolve_again,
            repair=repair,
        )

    return run


def as_llm_verify_seam(
    chat_fns,
    ledger,
    clock,
    cfg: dict | None = None,
) -> Callable[[Any, str, str], bool]:
    """Adapt CP4 to the head's narrow ``LlmVerifyFn = (plan, summary, matrix) -> bool``.

    The head iterates candidates and asks "keep this one?" per candidate. We map the CP4
    verdict for the winner candidate: ``confirm`` -> True (keep); ``runner_up``/``neither``
    -> False (demote, let the head walk to the next candidate). The missed-constraint /
    re-resolve path cannot be expressed through this bool seam and is degraded away — see
    the module docstring's seam-gap note. ``plan.question_raw`` supplies the question and
    ``plan.notes`` the ambiguity flags; the head passes the winner summary + matrix text.
    """
    cfg = cfg or {}
    chat = _resolve_chat(chat_fns)
    repair = cfg.get("repair")

    def seam(plan: Any, candidate_summary_text: str, pass_matrix_text: str) -> bool:
        question = getattr(plan, "question_raw", "") or ""
        notes = getattr(plan, "notes", "") or ""
        prompt = build_prompt(question, candidate_summary_text, pass_matrix_text, "(none)", notes)
        raw = guarded_call(
            CHECKPOINT_NAME, ledger, clock, lambda: chat(prompt), tier="text"
        )
        if raw is None:
            return True  # unavailable -> keep the deterministic winner
        obj, _errors = schemas.parse_with_repair(raw, schemas.validate_verification, repair)
        if obj is None:
            return True  # malformed -> keep the deterministic winner
        return obj["verdict"] == "confirm"

    return seam


# --------------------------------------------------------------------------- internals


def _resolve_chat(chat_fns) -> Callable[[list[dict[str, str]]], str]:
    if callable(chat_fns) and not hasattr(chat_fns, "chat"):
        return chat_fns
    for attr in ("chat", "text", "text_chat"):
        fn = getattr(chat_fns, attr, None)
        if callable(fn):
            return fn
    if isinstance(chat_fns, dict):
        for key in ("chat", "text"):
            if callable(chat_fns.get(key)):
                return chat_fns[key]
    if callable(chat_fns):
        return chat_fns
    raise TypeError("chat_fns does not expose a text ChatFn")


def _safe_remaining(fn: Callable[[], float]) -> float:
    try:
        return float(fn())
    except Exception:  # noqa: BLE001
        return float("inf")
