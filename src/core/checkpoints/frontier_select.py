"""CP5 — frontier selection (vision, rare).

Design doc §CP5: fires only when the scene proved multi-room (>=2 disconnected explored
regions or area over threshold) AND the robot is ahead of budget AND the ledger allows
(cap 1). Show the VLM the current panorama with numbered discs at the top-5 frontier
directions and the question; the VLM picks which direction to explore.

Contract: ``{"choice": 1-N, "reason": str}`` (N = number of rendered frontiers, default 5).

Outcome semantics:
* valid choice -> "choice": the caller drives toward ``frontiers[choice-1]``.
* skip / timeout / malformed / out-of-range -> "fallback": the caller uses its geometric
  top frontier (deterministic fallback).

The choice is 1-indexed in the contract (matching the rendered disc numbers); the outcome
exposes both the 1-indexed ``choice`` and a 0-indexed ``index`` for direct list access.

Vision input: the numbered panorama is encoded via an injectable ``encode_fn`` (default raw
.npy). Frontier centroids are projected to panorama columns by the caller (via
``tiling.azimuth_to_column``) before rendering — CP5 only consumes the finished image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.checkpoints import schemas
from core.checkpoints._runtime import guarded_call
from core.checkpoints._vision import default_encode_fn

CHECKPOINT_NAME = "frontier_select"

DEFAULT_N_FRONTIERS: int = 5

SYSTEM_PROMPT = (
    "You direct a robot exploring a multi-room space to answer a question. Numbered discs "
    "on the panorama mark unexplored directions. Reply with STRICT JSON only."
)

PROMPT_TEMPLATE = """\
Robot exploring to answer: {question}
Numbered directions (1-{n}) mark unexplored areas. Pick the direction most likely to help \
answer the question. Reply JSON only: {{"choice": 1-{n}, "reason": str}}"""


@dataclass(frozen=True)
class FrontierOutcome:
    """Result of one CP5 run.

    action:  "choice" (drive toward the picked frontier) | "fallback" (geometric top).
    choice:  1-indexed disc number the VLM picked (None on fallback).
    index:   0-indexed list position for ``frontiers[index]`` (None on fallback).
    reason:  the VLM's stated reason (audit).
    """

    action: str
    choice: int | None = None
    index: int | None = None
    reason: str = ""


def build_prompt(question: str, n_frontiers: int) -> list[dict[str, str]]:
    """Assemble the CP5 message from the question + number of rendered frontiers."""
    n = max(n_frontiers, 1)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(question=question or "(none)", n=n),
        },
    ]


def run_frontier_select(
    vision_chat: Callable[[list[dict[str, str]], list[bytes]], str],
    ledger,
    clock,
    *,
    question: str,
    panorama: Any,
    n_frontiers: int = DEFAULT_N_FRONTIERS,
    encode_fn: Callable[[Any], bytes] | None = None,
    repair: Callable[[str, list], str] | None = None,
) -> FrontierOutcome:
    """Run one ledger-gated, timeout-bounded CP5 selection and classify the choice."""
    n = max(n_frontiers, 1)
    encode = encode_fn or default_encode_fn
    images = [encode(panorama)] if panorama is not None else []
    prompt = build_prompt(question, n)
    reply = guarded_call(
        CHECKPOINT_NAME,
        ledger,
        clock,
        lambda: vision_chat(prompt, images),
        tier="vision",
    )
    if reply is None:
        return FrontierOutcome("fallback")

    obj, _errors = schemas.parse_with_repair(
        reply, lambda d: schemas.validate_frontier_select(d, n_choices=n), repair
    )
    if obj is None:
        return FrontierOutcome("fallback")

    choice = int(obj["choice"])
    return FrontierOutcome("choice", choice=choice, index=choice - 1, reason=obj.get("reason", ""))


def build_frontier_select(
    chat_fns,
    ledger,
    clock,
    cfg: dict | None = None,
) -> Callable[..., FrontierOutcome]:
    """Build the CP5 selector bound to a VisionChatFn, ledger, clock, config.

    ``cfg`` keys: ``n_frontiers`` (default 5), ``encode_fn`` (default raw .npy), ``repair``.
    Returns ``run(question, panorama, n_frontiers=None) -> FrontierOutcome``.
    """
    cfg = cfg or {}
    vision_chat = _resolve_vision(chat_fns)
    encode_fn = cfg.get("encode_fn") or default_encode_fn
    repair = cfg.get("repair")
    default_n = int(cfg.get("n_frontiers", DEFAULT_N_FRONTIERS))

    def run(question: str, panorama: Any = None, n_frontiers: int | None = None) -> FrontierOutcome:
        return run_frontier_select(
            vision_chat,
            ledger,
            clock,
            question=question,
            panorama=panorama,
            n_frontiers=default_n if n_frontiers is None else n_frontiers,
            encode_fn=encode_fn,
            repair=repair,
        )

    return run


def _resolve_vision(chat_fns) -> Callable[[list[dict[str, str]], list[bytes]], str]:
    for attr in ("vision_chat", "vision"):
        fn = getattr(chat_fns, attr, None)
        if callable(fn):
            return fn
    if isinstance(chat_fns, dict):
        for key in ("vision_chat", "vision"):
            if callable(chat_fns.get(key)):
                return chat_fns[key]
    if callable(chat_fns):
        return chat_fns
    raise TypeError("chat_fns does not expose a VisionChatFn")
