"""Deterministic post-processing for the LLM parse tiers' Plan output.

The local (and, defensively, any) LLM tier reliably matches the schema (it validates)
but sometimes gets the *semantics* wrong in three specific, systematically-reproducible
ways the deterministic regex floor never does (confirmed by direct re-invocation
against the live local model — see ``reports/local_llm_phase2/parse_battery.md``
"Divergence analysis" and issues #47/#48/#49):

* #47 -- a non-terminal "go near X" / "stop by X" route leg gets tagged ``via_near``
  instead of ``goto`` (rule 6 reserves ``via_near`` for "take the path near X" / "pass
  by X" phrasing only). The model also sometimes SPLITS a terminal "stop by X <relative
  clause>" into a spurious ``via_near`` leg carrying the disambiguator plus a bare
  ``goto`` duplicate of the same anchor as the real terminal leg.
* #48 -- a target with two chained relative clauses ("X on Y closest to Z") comes back
  as two flat peer clauses on the target instead of the second nested inside the
  first's anchor disambiguator (rule 4). The regex floor already encodes exactly when
  a trailing superlative should surface to the target instead of nesting (a bare, no
  relative-pronoun "on the Y closest to Z" on an OBJECT_REFERENCE question — see
  ``regex_tier._split_trailing_superlative``); this module mirrors that same
  decision so the two tiers agree on the ambiguous case and only fix the
  unambiguous one (rule-4 violations with an explicit "that is"/"that has" binder).
* #49 -- a spurious ``avoid`` entry gets fabricated by pairing two anchors that
  belong to different, unrelated disambiguators elsewhere in the Plan, on a question
  that contains no "avoid"/"without" language at all.
* #65 -- "stop at X between Y and Z" (rule 6's explicit exception: a goto leg whose
  anchor X carries a ``between`` disambiguator, NOT a corridor -- see prompts.py's
  own worked-example NOTE) sometimes comes back as a spurious ``corridor_between``
  leg (anchors Y, Z) immediately followed by a bare terminal ``goto`` leg (anchor X)
  instead of one ``goto`` leg with the disambiguator correctly nested. Distinguished
  from a genuine corridor leg (also a non-terminal ``corridor_between`` leg followed
  by a ``goto``, e.g. "take the path between the sofa and the tables, and stop at
  the cabinet") purely by the surface marker immediately before "between": a real
  corridor is introduced by "take the path between"/"go between"; the misplaced-
  disambiguator case has no such marker and instead has the terminal goto anchor's
  own noun immediately before "between".

Every rule here only ever *removes or reshapes* structure the model invented; it never
adds anchors, nouns, or predicates the model didn't already emit. Applied only to the
successfully-validated Plan a chat tier returns (before ``_stamp``), never to the
regex floor's own output.
"""
from __future__ import annotations

import re

from core.interfaces import QType
from core.plan_schema import Clause, LegKind, Plan, Pred, RouteLeg

# --------------------------------------------------------------------- #47: route legs

#: Surface markers that genuinely signal a "pass near / take the path near" leg (rule 6).
#: Absent any of these near a non-terminal `via_near` leg's anchor mention, the leg is a
#: mislabeled "go near X" / "stop by X" (both -> "goto" per rule 6).
_PASS_BY_MARKERS: tuple[str, ...] = (
    "pass by", "passing by", "path near", "path by", "take the path",
)


def _find_anchor_span(qlower: str, cursor: int, noun: str, raw: str) -> tuple[int, int]:
    """Find the next mention of an anchor (by raw, falling back to noun) at/after cursor.

    Returns (-1, -1) if neither surface form appears anywhere at/after cursor — the
    caller then treats "rest of the question" as the search window, so an unlocatable
    anchor still gets rule 6's more common default (goto) rather than blocking on it.
    """
    for candidate in (raw, noun):
        candidate = (candidate or "").strip().lower()
        if not candidate:
            continue
        pos = qlower.find(candidate, cursor)
        if pos >= 0:
            return pos, pos + len(candidate)
    return -1, -1


def _same_noun(a: str, b: str) -> bool:
    """Loose canonical-noun match: exact, or equal after stripping a trailing plural 's'."""
    a, b = (a or "").strip().lower(), (b or "").strip().lower()
    if a == b:
        return True
    return a.rstrip("s") == b.rstrip("s") and a and b


def _normalize_route_legs(plan: Plan, qtext: str) -> None:
    """Fix #47: downgrade non-terminal `via_near` legs lacking pass-by/path phrasing to
    `goto`, then collapse a downgraded leg immediately followed by a same-anchor `goto`
    leg (the "split terminal stop" degenerate case) into one `goto` leg.
    """
    if not plan.route:
        return
    qlower = qtext.lower()
    cursor = 0
    downgraded = [False] * len(plan.route)

    for i, leg in enumerate(plan.route):
        if not leg.anchors:
            continue
        if leg.kind is LegKind.CORRIDOR_BETWEEN:
            for a in leg.anchors:
                start, end = _find_anchor_span(qlower, cursor, a.noun, a.raw)
                if end >= 0:
                    cursor = end
            continue

        anchor = leg.anchors[0]
        start, end = _find_anchor_span(qlower, cursor, anchor.noun, anchor.raw)
        window_end = start if start >= 0 else len(qlower)
        is_terminal = i == len(plan.route) - 1
        if leg.kind is LegKind.VIA_NEAR and not is_terminal:
            window = qlower[cursor:window_end]
            if not any(marker in window for marker in _PASS_BY_MARKERS):
                leg.kind = LegKind.GOTO
                downgraded[i] = True
        if end >= 0:
            cursor = end

    merged: list[RouteLeg] = []
    skip_next = False
    for i, leg in enumerate(plan.route):
        if skip_next:
            skip_next = False
            continue
        nxt = plan.route[i + 1] if i + 1 < len(plan.route) else None
        if (
            downgraded[i]
            and leg.kind is LegKind.GOTO
            and nxt is not None
            and nxt.kind is LegKind.GOTO
            and len(leg.anchors) == 1
            and len(nxt.anchors) == 1
            and _same_noun(leg.anchors[0].noun, nxt.anchors[0].noun)
        ):
            a, b = leg.anchors[0], nxt.anchors[0]
            keep = a if a.disambiguator is not None else b
            merged.append(RouteLeg(kind=LegKind.GOTO, anchors=[keep]))
            skip_next = True
        else:
            merged.append(leg)
    plan.route = merged


# ----------------------------------------------------------- #48: stacked target clauses

#: A relative-pronoun copula binder ("that is"/"that are"/"which is"/"it is", per
#: `regex_tier._RELCLAUSE_TAIL_RE` / `_CONNECTOR_RE`) that ties a trailing superlative to
#: the immediately preceding anchor rather than the head target.
_RELATIVE_PRONOUN_RE = re.compile(r"\b(?:that|which|it)\s+(?:is|are)\b", re.IGNORECASE)

_SUPERLATIVE_PREDS = (Pred.CLOSEST_TO, Pred.FARTHEST_FROM)


def _normalize_stacked_target_clauses(plan: Plan, qtext: str) -> None:
    """Fix #48: nest a flat trailing clause into the first clause's anchor disambiguator,
    mirroring the regex floor's own nest-vs-surface rule (`regex_tier.
    _split_trailing_superlative`) so both tiers agree on the one case that's genuinely
    ambiguous (a bare, no-relative-pronoun trailing superlative on an OBJECT_REFERENCE
    question ranks the target itself, not the anchor) and only fix the unambiguous one.
    """
    if plan.target is None or len(plan.target.clauses) != 2:
        return
    c0, c1 = plan.target.clauses
    if len(c0.anchors) != 1 or c0.anchors[0].disambiguator is not None:
        return
    if len(c1.anchors) != 1:
        return

    should_nest = True
    if (
        plan.qtype is QType.OBJECT_REFERENCE
        and c1.pred in _SUPERLATIVE_PREDS
        and c0.pred is not Pred.WITH
        and not _RELATIVE_PRONOUN_RE.search(qtext)
    ):
        should_nest = False  # bare trailing superlative: floor surfaces it too, leave flat

    if should_nest:
        c0.anchors[0].disambiguator = c1
        plan.target.clauses = [c0]


# ------------------------------------------------------- #65: misplaced between-disambiguator

#: Surface markers that genuinely introduce a corridor leg (mirrors regex_tier._CORRIDOR_KEYS
#: plus its "path between" variant) -- rule 6's ONLY legitimate `corridor_between` triggers.
_CORRIDOR_MARKERS: tuple[str, ...] = ("take the path between", "go between", "path between")

#: How far back from a "between" occurrence to look for a corridor marker / the terminal
#: goto anchor's surface form. Generous enough for "take the path between the sofa and",
#: short enough not to accidentally reach across an unrelated earlier clause.
_BETWEEN_LOOKBACK = 40


def _merge_misplaced_between_disambiguator(plan: Plan, qtext: str) -> None:
    """Fix #65: merge a `corridor_between` leg into an immediately-following bare `goto`
    leg when the text shows this was really "stop at X between Y and Z" (rule 6's
    between-disambiguator exception), not a genuine corridor leg. See module docstring.
    """
    if len(plan.route) < 2:
        return
    qlower = qtext.lower()
    merged: list[RouteLeg] = []
    i = 0
    while i < len(plan.route):
        leg = plan.route[i]
        nxt = plan.route[i + 1] if i + 1 < len(plan.route) else None
        if (
            leg.kind is LegKind.CORRIDOR_BETWEEN
            and len(leg.anchors) == 2
            and nxt is not None
            and nxt.kind is LegKind.GOTO
            and len(nxt.anchors) == 1
            and nxt.anchors[0].disambiguator is None
        ):
            goto_anchor = nxt.anchors[0]
            candidate = (goto_anchor.raw or goto_anchor.noun or "").strip().lower()
            if candidate and _anchor_precedes_bare_between(qlower, candidate):
                goto_anchor.disambiguator = Clause(pred=Pred.BETWEEN, anchors=leg.anchors)
                merged.append(RouteLeg(kind=LegKind.GOTO, anchors=[goto_anchor]))
                i += 2
                continue
        merged.append(leg)
        i += 1
    plan.route = merged


def _anchor_precedes_bare_between(qlower: str, candidate: str) -> bool:
    """True if `candidate` (the terminal goto anchor's surface form) sits immediately
    before some "between" occurrence in the text, with no corridor marker in that same
    lookback window -- the "the vase between the TV and the door" lexical signature.
    """
    pos = 0
    while True:
        bpos = qlower.find("between", pos)
        if bpos < 0:
            return False
        window = qlower[max(0, bpos - _BETWEEN_LOOKBACK):bpos]
        if candidate in window and not any(m in window for m in _CORRIDOR_MARKERS):
            return True
        pos = bpos + len("between")


# ------------------------------------------------------------------- #49: fabricated avoid

_AVOID_MARKERS: tuple[str, ...] = ("avoid", "without")


def _drop_fabricated_avoid(plan: Plan, qtext: str) -> None:
    """Fix #49: a question with no "avoid"/"without" language can never legitimately carry
    an `avoid` entry (rule 7) -- drop any the model fabricated and log it in `notes`
    rather than silently discarding evidence of the defect.
    """
    if not plan.avoid:
        return
    qlower = qtext.lower()
    if any(marker in qlower for marker in _AVOID_MARKERS):
        return
    n = len(plan.avoid)
    plan.avoid = []
    note = f"dropped {n} fabricated avoid entr{'y' if n == 1 else 'ies'}: no avoid/without language in question"
    plan.notes = f"{plan.notes} | {note}" if plan.notes else note


# --------------------------------------------------------------------------- entry point


def normalize_llm_plan(plan: Plan, qtext: str) -> Plan:
    """Apply all four LLM-tier normalization rules in place; returns `plan` for chaining."""
    _normalize_route_legs(plan, qtext)
    _merge_misplaced_between_disambiguator(plan, qtext)
    _normalize_stacked_target_clauses(plan, qtext)
    _drop_fabricated_avoid(plan, qtext)
    return plan
