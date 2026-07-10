"""CP2 — detector-miss recovery (vision).

Design doc §CP2: a plan-critical noun has 0 instances after the coverage threshold
(explore budget >=60% spent) — at most once per question (ledger cap 1). Show the VLM the
4 current tiles (downscaled) plus the noun + its raw surface form and ask whether any tile
contains it.

Contract: ``{"present": bool, "tile": 0-3|null, "bbox_hint": [x1,y1,x2,y2]|null,
"confidence": 0-1}``.

Outcome semantics:
* present True with a usable tile+bbox -> "provisional": the caller casts ``bbox_hint``
  through fusion (min_points relaxed to 3) into a provisional instance
  (n_obs=1, score=confidence*0.5) and re-navigates toward it.
* present False / skip / timeout / malformed -> "absent": proceed to the resolve fallback
  ladder unchanged (deterministic fallback).

Hallucination cap (design "Risk note"): the reported ``score`` is capped so a provisional
instance can never on its own satisfy the >=3-obs early-answer gate — enforced here by
returning ``n_obs=1`` and ``score = confidence * PROVISIONAL_SCORE_FACTOR``.

Vision input: the 4 tiles are encoded via an injectable ``encode_fn`` (default raw .npy);
one image per tile, tile order preserved so ``tile`` indexes the returned list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from core.checkpoints import schemas
from core.checkpoints._runtime import guarded_call
from core.checkpoints._vision import default_encode_fn

CHECKPOINT_NAME = "miss_recovery"

#: Design "Risk note": provisional score factor keeps a hallucinated hit below the early gate.
PROVISIONAL_SCORE_FACTOR: float = 0.5
PROVISIONAL_N_OBS: int = 1
#: min_points relaxed for provisional-instance fusion (design: "min_points relaxed to 3").
PROVISIONAL_MIN_POINTS: int = 3

SYSTEM_PROMPT = (
    "You assist a robot searching a room. Given four camera tiles, report whether a named "
    "object appears in any tile. Reply with STRICT JSON only."
)

PROMPT_TEMPLATE = """\
Does any of these {n} tiles contain a {noun} (also written '{raw}')?
Tiles are numbered 0-{last} in order. Reply JSON only: \
{{"present": bool, "tile": 0-{last}|null, "bbox_hint": [x1,y1,x2,y2] in tile pixels|null, \
"confidence": 0-1}}"""


@dataclass(frozen=True)
class MissRecoveryOutcome:
    """Result of one CP2 run.

    action:  "provisional" (fuse a provisional instance + re-navigate) | "absent".
    tile:    which tile (0-3) the hit is in (None when absent).
    bbox_hint: [x1,y1,x2,y2] tile-pixel box (None when absent/unspecified).
    confidence: raw model confidence.
    n_obs / score / min_points: parameters for the provisional-instance fusion, capped so
        the hit cannot satisfy the >=3-obs early-answer gate on its own.
    """

    action: str
    tile: int | None = None
    bbox_hint: list[float] | None = None
    confidence: float = 0.0
    n_obs: int = PROVISIONAL_N_OBS
    score: float = 0.0
    min_points: int = PROVISIONAL_MIN_POINTS


def build_prompt(noun: str, raw: str, n_tiles: int) -> list[dict[str, str]]:
    """Assemble the CP2 message from the missed noun + its raw surface form."""
    last = max(n_tiles - 1, 0)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(
                n=n_tiles, noun=noun or "object", raw=raw or noun or "object", last=last
            ),
        },
    ]


def run_miss_recovery(
    vision_chat: Callable[[list[dict[str, str]], list[bytes]], str],
    ledger,
    clock,
    *,
    noun: str,
    raw: str,
    tiles: Sequence[Any],
    encode_fn: Callable[[Any], bytes] | None = None,
    repair: Callable[[str, list], str] | None = None,
) -> MissRecoveryOutcome:
    """Run one ledger-gated, timeout-bounded CP2 recovery and classify the verdict."""
    encode = encode_fn or default_encode_fn
    images = [encode(t) for t in tiles]
    prompt = build_prompt(noun, raw, len(tiles))
    reply = guarded_call(
        CHECKPOINT_NAME,
        ledger,
        clock,
        lambda: vision_chat(prompt, images),
        tier="vision",
    )
    if reply is None:
        return MissRecoveryOutcome("absent")

    obj, _errors = schemas.parse_with_repair(reply, schemas.validate_miss_recovery, repair)
    if obj is None:
        return MissRecoveryOutcome("absent")

    if not obj["present"]:
        return MissRecoveryOutcome("absent", confidence=float(obj["confidence"]))

    conf = float(obj["confidence"])
    tile = obj.get("tile")
    if isinstance(tile, int) and 0 <= tile < len(tiles):
        chosen = tile
    else:
        chosen = None  # present but no usable tile pointer
    bbox = obj.get("bbox_hint")
    if bbox is None or chosen is None:
        # "present" but no fusable localisation -> nothing to navigate toward; treat as absent.
        return MissRecoveryOutcome("absent", tile=chosen, confidence=conf)
    return MissRecoveryOutcome(
        "provisional",
        tile=chosen,
        bbox_hint=[float(x) for x in bbox],
        confidence=conf,
        n_obs=PROVISIONAL_N_OBS,
        score=conf * PROVISIONAL_SCORE_FACTOR,
        min_points=PROVISIONAL_MIN_POINTS,
    )


def build_miss_recovery(
    chat_fns,
    ledger,
    clock,
    cfg: dict | None = None,
) -> Callable[..., MissRecoveryOutcome]:
    """Build the CP2 recoverer bound to a VisionChatFn, ledger, clock, config.

    ``cfg`` keys: ``encode_fn`` (default raw .npy), ``repair``.
    Returns ``run(noun, raw, tiles) -> MissRecoveryOutcome``.
    """
    cfg = cfg or {}
    vision_chat = _resolve_vision(chat_fns)
    encode_fn = cfg.get("encode_fn") or default_encode_fn
    repair = cfg.get("repair")

    def run(noun: str, raw: str, tiles: Sequence[Any]) -> MissRecoveryOutcome:
        return run_miss_recovery(
            vision_chat,
            ledger,
            clock,
            noun=noun,
            raw=raw,
            tiles=tiles,
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
