"""CP3 — anchor confirmation on arrival (vision, IF only).

Design doc §CP3: on arrival at each IF sub-goal (ledger cap 3), show the VLM the tile crop
centred on the anchor's projected image location and ask whether the centred object really
is the anchor the map grounded. On a confident mismatch, demote and re-plan to the
runner-up; otherwise continue on the map's belief (one mismatch report is weaker than the
tracked map). Never blocks the drive.

Contract: ``{"match": bool, "actual_label": str|null, "confidence": 0-1}``.

Outcome semantics (OR-F9):
* match True  (any conf)                          -> "confirm"  (continue).
* match False, confidence <0.8                    -> "confirm"  (too weak to override map).
* match False, actual_label a SYNONYM/same class  -> "confirm"  (couch==sofa is not a
    mismatch — routed through the ``core.perception.vocab`` bridge).
* match False, actual_label null / empty          -> "confirm"  (no named alternative).
* match False, actual_label a DIFFERENT class, confidence >=0.8 -> "demote"
    (caller demotes + re-resolves this anchor).
* skip / timeout / malformed                       -> "confirm" (deterministic fallback).

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
from core.perception.vocab import bridged_agree  # read-only: same-class vocab bridge

CHECKPOINT_NAME = "anchor_confirm"

#: OR-F9: a mismatch only overrides the map at high confidence. VLM self-reported
#: confidences cluster >=0.8, so the old 0.6 floor demoted on essentially every mismatch;
#: 0.8 keeps the "map beats one report" rule the design text states.
MISMATCH_MIN_CONFIDENCE: float = 0.8

SYSTEM_PROMPT = (
    "You confirm whether the object centred in a robot camera crop matches what the "
    "robot's map believes it is. Reply with STRICT JSON only."
)

PROMPT_TEMPLATE = """\
First name the object centred in this crop (actual_label), then judge whether it matches: \
the robot believes it is a {anchor}. If the crop is ambiguous, blurred, or the object is \
cut off, answer match=true with low confidence.
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
    anchor_noun: str | None = None,
    encode_fn: Callable[[Any], bytes] | None = None,
    repair: Callable[[str, list], str] | None = None,
) -> AnchorConfirmOutcome:
    """Run one ledger-gated, timeout-bounded CP3 confirmation and classify the verdict.

    ``anchor_noun`` (OR-F9) is the canonical anchor noun used to test whether a mismatch's
    ``actual_label`` is a genuinely different class (via the vocab bridge). Defaults to
    ``anchor_desc`` when not supplied.
    """
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
    if _should_demote(match, conf, actual, anchor_noun or anchor_desc):
        return AnchorConfirmOutcome("demote", match, actual, conf)
    return AnchorConfirmOutcome("confirm", match, actual, conf)


def _should_demote(match: bool, conf: float, actual_label: str | None, anchor_noun: str) -> bool:
    """OR-F9 demote gate: demote ONLY when the VLM reports match=false at high confidence
    AND names a genuinely different object class.

    A synonym / same-class ``actual_label`` (couch vs sofa) is NOT a mismatch — routed
    through :func:`core.perception.vocab.bridged_agree`. A null/empty ``actual_label`` gives
    no named alternative, so we keep the map's belief.
    """
    if match or conf < MISMATCH_MIN_CONFIDENCE:
        return False
    label = (actual_label or "").strip()
    if not label:
        return False  # no named alternative -> trust the map
    if bridged_agree(anchor_noun, label):
        return False  # same class under the vocab bridge (synonym) -> confirm
    return True


def build_anchor_confirm(
    chat_fns,
    ledger,
    clock,
    cfg: dict | None = None,
) -> Callable[..., AnchorConfirmOutcome]:
    """Build the CP3 confirmer bound to a VisionChatFn, ledger, clock, config.

    ``cfg`` keys: ``encode_fn`` (image -> bytes; default raw .npy), ``repair``.
    Returns ``run(anchor_desc, crop, anchor_noun=None) -> AnchorConfirmOutcome``.
    """
    cfg = cfg or {}
    vision_chat = _resolve_vision(chat_fns)
    encode_fn = cfg.get("encode_fn") or default_encode_fn
    repair = cfg.get("repair")

    def run(
        anchor_desc: str, crop: Any = None, anchor_noun: str | None = None
    ) -> AnchorConfirmOutcome:
        return run_anchor_confirm(
            vision_chat,
            ledger,
            clock,
            anchor_desc=anchor_desc,
            crop=crop,
            anchor_noun=anchor_noun,
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
