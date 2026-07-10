"""CP3 — anchor confirmation on arrival (vision, IF only).

Design doc §CP3: on arrival at each IF sub-goal (ledger cap 3), show the VLM the tile crop
centred on the anchor's projected image location and ask whether the centred object really
is the anchor the map grounded. On a confident mismatch, demote and re-plan to the
runner-up; otherwise continue on the map's belief (one mismatch report is weaker than the
tracked map). Never blocks the drive.

Contract: ``{"match": bool, "actual_label": str|null, "confidence": 0-1}``.

Outcome semantics:
* match True  (any conf)        -> "confirm"   (continue).
* match False, confidence <0.6  -> "confirm"   (too weak to override the map — design rule).
* match False, confidence >=0.6 -> "demote"    (caller demotes + re-resolves this anchor).
* skip / timeout / malformed    -> "confirm"   (deterministic fallback = trust the map).

Vision input: the anchor crop is passed to an injectable ``encode_fn(image) -> bytes``.
cv2/PIL are not deps, so the default encoder emits raw ``.npy`` bytes (test-friendly, no
image lib); real JPEG encoding is a Phase-2 wiring note (swap the encoder at build time).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.checkpoints import schemas
from core.checkpoints._runtime import guarded_call
from core.checkpoints._vision import default_encode_fn

CHECKPOINT_NAME = "anchor_confirm"

#: Design rule: a mismatch only overrides the map when the VLM is at least this confident.
MISMATCH_MIN_CONFIDENCE: float = 0.6

SYSTEM_PROMPT = (
    "You confirm whether the object centred in a robot camera crop matches what the "
    "robot's map believes it is. Reply with STRICT JSON only."
)

PROMPT_TEMPLATE = """\
The robot believes the object centred in this crop is a {anchor}.
Reply JSON only: {{"match": bool, "actual_label": str|null, "confidence": 0-1}}"""


@dataclass(frozen=True)
class AnchorConfirmOutcome:
    """Result of one CP3 run.

    action:  "confirm" (continue on the map) | "demote" (re-resolve/re-plan this anchor).
    match:   the raw model verdict (None when the call was skipped/failed).
    actual_label / confidence: audit fields.
    """

    action: str
    match: bool | None
    actual_label: str | None = None
    confidence: float = 0.0


def build_prompt(anchor_desc: str) -> list[dict[str, str]]:
    """Assemble the CP3 message from the anchor noun + attributes description."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT_TEMPLATE.format(anchor=anchor_desc or "object")},
    ]


def run_anchor_confirm(
    vision_chat: Callable[[list[dict[str, str]], list[bytes]], str],
    ledger,
    clock,
    *,
    anchor_desc: str,
    crop: Any,
    encode_fn: Callable[[Any], bytes] | None = None,
    repair: Callable[[str, list], str] | None = None,
) -> AnchorConfirmOutcome:
    """Run one ledger-gated, timeout-bounded CP3 confirmation and classify the verdict."""
    encode = encode_fn or default_encode_fn
    images = [encode(crop)] if crop is not None else []
    prompt = build_prompt(anchor_desc)
    raw = guarded_call(
        CHECKPOINT_NAME,
        ledger,
        clock,
        lambda: vision_chat(prompt, images),
        tier="vision",
    )
    if raw is None:
        return AnchorConfirmOutcome("confirm", None, None, 0.0)

    obj, _errors = schemas.parse_with_repair(raw, schemas.validate_anchor_confirm, repair)
    if obj is None:
        return AnchorConfirmOutcome("confirm", None, None, 0.0)

    match = bool(obj["match"])
    conf = float(obj["confidence"])
    actual = obj.get("actual_label")
    if not match and conf >= MISMATCH_MIN_CONFIDENCE:
        return AnchorConfirmOutcome("demote", match, actual, conf)
    return AnchorConfirmOutcome("confirm", match, actual, conf)


def build_anchor_confirm(
    chat_fns,
    ledger,
    clock,
    cfg: dict | None = None,
) -> Callable[..., AnchorConfirmOutcome]:
    """Build the CP3 confirmer bound to a VisionChatFn, ledger, clock, config.

    ``cfg`` keys: ``encode_fn`` (image -> bytes; default raw .npy), ``repair``.
    Returns ``run(anchor_desc, crop) -> AnchorConfirmOutcome``.
    """
    cfg = cfg or {}
    vision_chat = _resolve_vision(chat_fns)
    encode_fn = cfg.get("encode_fn") or default_encode_fn
    repair = cfg.get("repair")

    def run(anchor_desc: str, crop: Any = None) -> AnchorConfirmOutcome:
        return run_anchor_confirm(
            vision_chat,
            ledger,
            clock,
            anchor_desc=anchor_desc,
            crop=crop,
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
        return chat_fns  # a bare VisionChatFn
    raise TypeError("chat_fns does not expose a VisionChatFn")
