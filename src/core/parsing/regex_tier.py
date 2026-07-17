"""Deterministic regex-tier parser: question text -> plan_schema.Plan, total (never raises).

Last rung of the checkpoint-1 parse ladder. Handles the structure of all 75 training
questions (docs/question_analysis.md): qtype classification from bare text, noun/attribute
extraction via core.parsing.vocab, relation-phrase -> Pred mapping with right-branching
nesting, ordered leg segmentation and avoid-clause extraction for instruction-following.
Novel phrasing degrades to a structurally valid, best-effort Plan with a note.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

from core.interfaces import QType
from core.parsing import vocab
from core.plan_schema import Anchor, AvoidSpec, Clause, LegKind, Plan, Pred, RouteLeg, TargetSpec

# --------------------------------------------------------------------- qtype classification

_NUMERICAL_RE = re.compile(r"^\s*how\s+many\b|^\s*count\b|\bcount\s+the\s+number\b", re.I)
_INSTRUCTION_RE = re.compile(
    r"\b(go\s+to|go\s+near|go\s+between|go\s+past|take\s+the\s+path|stop\s+at|stop\s+by"
    r"|pass\s+by|navigate|walk\s+to|head\s+to|move\s+to)\b",
    re.I,
)


def classify_qtype(text: str) -> QType:
    """Classify bare question text into a QType (numerical checked first, then movement verbs)."""
    if _NUMERICAL_RE.search(text):
        return QType.NUMERICAL
    if _INSTRUCTION_RE.search(text):
        return QType.INSTRUCTION_FOLLOWING
    return QType.OBJECT_REFERENCE


# --------------------------------------------------------------------- relation lexicon

# Ordered alternation: longest / most specific tokens first.
_REL_TOKENS: list[tuple[str, Pred | None]] = [
    (r"that\s+has", Pred.WITH),  # 'that has X on it' == 'with X on it'
    (r"closest\s+to", Pred.CLOSEST_TO),
    (r"nearest\s+to", Pred.CLOSEST_TO),
    (r"farthest\s+from", Pred.FARTHEST_FROM),
    (r"furthest\s+from", Pred.FARTHEST_FROM),
    (r"farthest\s+away\s+from", Pred.FARTHEST_FROM),
    # DD-A5: near-synonyms ("next to", "beside", "adjacent to", "close to") are
    # generated with the `near` threshold, so they route to Pred.NEAR — not the
    # tighter Pred.NEXT_TO, which nothing routes to any more.
    (r"next\s+to", Pred.NEAR),
    (r"adjacent\s+to", Pred.NEAR),
    (r"close\s+to", Pred.NEAR),
    (r"on\s+top\s+of", Pred.ON),
    (r"between", Pred.BETWEEN),
    (r"near", Pred.NEAR),
    (r"above", Pred.ABOVE),
    (r"below", Pred.UNDER),
    (r"underneath", Pred.UNDER),
    (r"under", Pred.UNDER),
    (r"beside", Pred.NEAR),
    (r"inside", Pred.IN),
    (r"with", Pred.WITH),
    (r"on", Pred.ON),
    (r"in", Pred.IN),
]

_REL_ALTERNATION = "|".join(pat for pat, _ in _REL_TOKENS)

# Superlative predicates ("closest to"/"farthest from" family) and the alternation of
# their surface tokens — used to surface a *trailing* superlative to the head target
# (see `_split_trailing_superlative`).
# Canonical superlative-predicate set. core.groundtruth.scoring imports this to derive
# its _SUPERLATIVE_PRED_VALUES (value-strings) — keep this the single source of truth.
_SUPERLATIVE_PREDS = {Pred.CLOSEST_TO, Pred.FARTHEST_FROM}
_SUPERLATIVE_ALTERNATION = "|".join(
    pat for pat, pred in _REL_TOKENS if pred in _SUPERLATIVE_PREDS
)
_SUPERLATIVE_AT_START_RE = re.compile(rf"^({_SUPERLATIVE_ALTERNATION})\b")
_SUPERLATIVE_SEARCH_RE = re.compile(rf"\b({_SUPERLATIVE_ALTERNATION})\b")
# A relative-clause connector ("that is"/"which are") right before a superlative binds
# it to the preceding anchor, not the head target — the superlative is NOT surfaced.
_RELCLAUSE_TAIL_RE = re.compile(r"\b(that|which)\s+(is|are)\s*$")

# Boundary between a noun phrase and its trailing relation chain ('is/are/that is' included).
_REL_BOUNDARY_RE = re.compile(
    rf"\b(that\s+is|that\s+are|which\s+is|is|are|{_REL_ALTERNATION})\b"
)
_REL_AT_START_RE = re.compile(rf"^({_REL_ALTERNATION})\b")
_CONNECTOR_RE = re.compile(r"^(?:that\s+is|that\s+are|which\s+is|it\s+is|is|are)\b[\s,]*")
_PRONOUN_TAIL_RE = re.compile(r"\s*\b(on|above|below|under)\s+(it|them)\b")


def _pred_of(token: str) -> Pred:
    """Map a matched relation token (whitespace-normalised) to its Pred."""
    norm = re.sub(r"\s+", " ", token.strip())
    for pat, pred in _REL_TOKENS:
        if re.fullmatch(pat, norm) and pred is not None:
            return pred
    raise ValueError(f"unmapped relation token: {token!r}")


# --------------------------------------------------------------------- parse context


@dataclass
class _Ctx:
    """Accumulates audit notes during one parse."""

    notes: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)


def _normalize(text: str) -> str:
    """Lowercase, strip terminal punctuation, collapse whitespace."""
    t = text.strip().lower()
    t = t.rstrip(" .?!")
    return re.sub(r"\s+", " ", t)


def _clean_segment(text: str) -> str:
    """Strip leading/trailing connector words and punctuation from a leg/clause segment."""
    words = [w for w in re.split(r"\s+", text.strip(" ,.;")) if w]
    drop = {"and", "then", "first", "finally", ",", "."}
    while words and words[0].strip(",.") in drop:
        words.pop(0)
    while words and words[-1].strip(",.") in drop:
        words.pop()
    return " ".join(words).strip(" ,.;")


# --------------------------------------------------------------------- NP / clause grammar


def _split_at_relation(text: str) -> tuple[str, str]:
    """Split 'NP <relation chain>' at the first relation/connector boundary."""
    m = _REL_BOUNDARY_RE.search(text)
    if m is None:
        return text, ""
    if not text[: m.start()].strip(" ,."):
        return text, ""  # boundary at start: whole text is the NP (degenerate)
    return text[: m.start()], text[m.start() :]


def _parse_anchor(text: str, ctx: _Ctx) -> Anchor:
    """Parse one anchor noun phrase plus its (recursive) relation-chain disambiguator."""
    s = _clean_segment(text)
    np_part, rel_part = _split_at_relation(s)
    noun, attrs, raw, indefinite = vocab.match_noun(np_part.split())
    if indefinite:
        ctx.note(f"indefinite article on '{raw}' (any matching instance qualifies)")
    dis = _parse_clause(rel_part, ctx) if rel_part else None
    return Anchor(noun=noun, raw=raw, attributes=attrs, disambiguator=dis)


def _parse_clause(text: str, ctx: _Ctx) -> Clause | None:
    """Parse a relation chain starting at a relation token into one (nested) Clause."""
    s = text.strip(" ,.;")
    while True:  # strip copula/relative-pronoun connectors
        m = _CONNECTOR_RE.match(s)
        if m is None or m.end() == 0:
            break
        s = s[m.end() :].lstrip(" ,")
    if not s:
        return None
    negated = False
    if s.startswith("not "):
        negated = True
        s = s[4:]
    m = _REL_AT_START_RE.match(s)
    if m is None:
        ctx.note(f"unparsed clause text dropped: '{s}'")
        return None
    pred = _pred_of(m.group(1))
    rest = s[m.end() :].strip(" ,.")
    if re.match(r"^(it|them)\b", rest):
        return None  # pronoun tail already consumed by a WITH handler upstream
    if pred is Pred.BETWEEN:
        anchors = _split_pair(rest, ctx, what="between")
        return Clause(pred=Pred.BETWEEN, anchors=anchors, negated=negated)
    if pred is Pred.WITH:
        tail = _PRONOUN_TAIL_RE.search(rest)
        if tail is not None:
            inner = rest[: tail.start()]
            rel_word = tail.group(1)
            # 'with X on it' -> WITH; 'with X above it' -> subject UNDER X; vice versa
            if rel_word == "above":
                pred = Pred.UNDER
            elif rel_word in ("below", "under"):
                pred = Pred.ABOVE
            leftover = rest[tail.end() :].strip(" ,.")
            if leftover:
                ctx.note(f"text after pronoun tail dropped: '{leftover}'")
            rest = inner
        return Clause(pred=pred, anchors=[_parse_anchor(rest, ctx)], negated=negated)
    return Clause(pred=pred, anchors=[_parse_anchor(rest, ctx)], negated=negated)


def _split_pair(text: str, ctx: _Ctx, what: str) -> list[Anchor]:
    """Parse 'A and B' or 'the two Xs' into exactly two anchors (duplicating on failure)."""
    s = _clean_segment(text)
    m = re.match(r"^(?:the\s+)?(?:two|both)\s+(.+)$", s)
    if m is not None:
        return [_parse_anchor(m.group(1), ctx), _parse_anchor(m.group(1), ctx)]
    parts = re.split(r"\s+and\s+", s, maxsplit=1)
    if len(parts) == 2:
        return [_parse_anchor(parts[0], ctx), _parse_anchor(parts[1], ctx)]
    ctx.note(f"{what} pair not found in '{s}'; anchor duplicated")
    a = _parse_anchor(s, ctx)
    return [a, copy.deepcopy(a)]


def _split_trailing_superlative(rel_part: str) -> tuple[str, str]:
    """Split a target relation chain 'on the Y closest to Z' into (head, superlative).

    A superlative that TRAILS an earlier relation ranks the head TARGET, not the
    intervening anchor: "the speaker on the tv cabinet closest to the potted plant"
    ranks the speakers by distance to the plant — it is not "the cabinet closest to
    the plant". The default right-branching NP grammar would nest 'closest to Z' as a
    disambiguator of Y; here we peel it off so it surfaces as a second top-level clause
    on the target. Returns ``(head, "")`` when there is nothing to surface — no
    superlative, or the superlative already *leads* the chain (then it is the target's
    own single relation and stays as one clause).
    """
    lead = rel_part.lstrip(" ,.;")
    m = _CONNECTOR_RE.match(lead)  # skip a leading copula/relative pronoun
    if m is not None:
        lead = lead[m.end() :].lstrip(" ,")
    if _SUPERLATIVE_AT_START_RE.match(lead):
        return rel_part, ""  # superlative already heads the chain — leave as one clause
    sm = _SUPERLATIVE_SEARCH_RE.search(rel_part)
    if sm is None or not rel_part[: sm.start()].strip(" ,.;"):
        return rel_part, ""  # no trailing superlative with preceding relation content
    head = rel_part[: sm.start()]
    # An explicit relative pronoun binds the superlative to the immediately preceding
    # anchor Y ("on the sofa THAT IS closest to Z" = the sofa closest to Z), NOT the
    # head target — only the *bare* "on the Y closest to Z" surfaces to the target.
    if _RELCLAUSE_TAIL_RE.search(head):
        return rel_part, ""
    return head, rel_part[sm.start() :]


def _parse_target(text: str, ctx: _Ctx, *, surface_superlative: bool = False) -> TargetSpec:
    """Parse '<NP> <relation chain>' into a TargetSpec (usually one top-level clause).

    When ``surface_superlative`` is set (object-reference selection), a trailing
    superlative ("... on the Y closest to Z") surfaces to the head target as a second
    top-level clause instead of nesting under the anchor Y — see
    :func:`_split_trailing_superlative`. Counting questions leave it OFF: a superlative
    cannot rank a *cardinality*, so there it disambiguates the anchor Y and must stay
    nested (else the count would drop the anchor filter and over-count).
    """
    np_part, rel_part = _split_at_relation(_clean_segment(text))
    noun, attrs, raw, indefinite = vocab.match_noun(np_part.split())
    if indefinite:
        ctx.note(f"indefinite article on target '{raw}'")
    clauses: list[Clause] = []
    if rel_part:
        head, superl = (
            _split_trailing_superlative(rel_part) if surface_superlative else (rel_part, "")
        )
        c = _parse_clause(head, ctx)
        if c is not None:
            clauses.append(c)
        if superl:
            sc = _parse_clause(superl, ctx)
            if sc is not None:
                clauses.append(sc)
                ctx.note("trailing superlative surfaced to head target")
    return TargetSpec(noun=noun, raw=raw, attributes=attrs, clauses=clauses)


# --------------------------------------------------------------------- per-qtype parsers

_HOW_MANY_RE = re.compile(r"^how\s+many\s+(.*)$")
_COUNT_RE = re.compile(r"^count\s+(?:the\s+)?(?:number\s+of\s+)?(.*)$")
_FIND_RE = re.compile(r"^(?:find|locate|identify|select)\s+(.*)$")


def _parse_numerical(text: str, ctx: _Ctx) -> TargetSpec:
    """Parse a counting question body into its TargetSpec."""
    for rx in (_HOW_MANY_RE, _COUNT_RE):
        m = rx.match(text)
        if m is not None:
            return _parse_target(m.group(1), ctx)
    ctx.note("no counting prefix found; parsing whole text as target")
    return _parse_target(text, ctx)


def _parse_object_reference(text: str, ctx: _Ctx) -> TargetSpec:
    """Parse a 'Find the ...' / bare-NP object-reference question into its TargetSpec."""
    m = _FIND_RE.match(text)
    return _parse_target(
        m.group(1) if m is not None else text, ctx, surface_superlative=True
    )


# ---- instruction-following ------------------------------------------------------

_LEG_RE = re.compile(
    r"\b(take\s+the\s+path\s+between|take\s+the\s+path\s+near|take\s+the\s+path\s+through"
    r"|go\s+between|go\s+to|go\s+near|go\s+past|pass\s+by|stop\s+at|stop\s+by"
    r"|walk\s+to|head\s+to|(?:then|finally)\s*,?\s+to)\b"
)
_AVOID_RE = re.compile(
    r"(?:,\s*)?\b(?:and\s+|while\s+)?avoid(?:ing)?\s+(?:the\s+)?path\s+(between|near)\s+"
)
_AVOID_PLAIN_RE = re.compile(r"(?:,\s*)?\b(?:and\s+|while\s+)?avoid(?:ing)?\s+")

_CORRIDOR_KEYS = {"take the path between", "go between"}
_VIA_KEYS = {"take the path near", "take the path through", "pass by", "go past"}


def _leg_kind(token: str) -> LegKind:
    """Map a leg-intro verb token to its LegKind."""
    norm = re.sub(r"\s+", " ", token.replace(",", " ").strip())
    if norm in _CORRIDOR_KEYS:
        return LegKind.CORRIDOR_BETWEEN
    if norm in _VIA_KEYS:
        return LegKind.VIA_NEAR
    return LegKind.GOTO


def _split_continuation(body: str) -> tuple[str, str]:
    """Split a corridor/via leg body at a bare ' to ' continuation ('...the bed to the picture...')."""
    masked = (
        body.replace("closest to", "closest\x00")
        .replace("next to", "next\x00")
        .replace("adjacent to", "adjacent\x00")
        .replace("close to", "close\x00")
    )
    parts = masked.split(" to ", 1)
    unmask = lambda s: s.replace("\x00", " to")  # noqa: E731
    if len(parts) == 2:
        return unmask(parts[0]), unmask(parts[1])
    return unmask(parts[0]), ""


def _extract_avoids(text: str, ctx: _Ctx) -> tuple[str, list[AvoidSpec]]:
    """Remove avoid clauses from instruction text, returning the cleaned text + AvoidSpecs."""
    avoids: list[AvoidSpec] = []
    while True:
        m = _AVOID_RE.search(text)
        if m is None:
            break
        spec = text[m.end() :]
        nxt = _LEG_RE.search(spec)
        trailing = ""
        if nxt is not None:
            trailing = spec[nxt.start() :]
            spec = spec[: nxt.start()]
        spec = _clean_segment(spec)
        if m.group(1) == "between":
            avoids.append(AvoidSpec(between=_split_pair(spec, ctx, what="avoid")))
        else:
            avoids.append(AvoidSpec(near=_parse_anchor(spec, ctx)))
        text = (text[: m.start()] + " " + trailing).strip()
    if not avoids:  # 'avoid the X' without 'the path' -> disc avoid, best effort
        m = _AVOID_PLAIN_RE.search(text)
        if m is not None:
            spec = text[m.end() :]
            nxt = _LEG_RE.search(spec)
            trailing = ""
            if nxt is not None:
                trailing = spec[nxt.start() :]
                spec = spec[: nxt.start()]
            avoids.append(AvoidSpec(near=_parse_anchor(_clean_segment(spec), ctx)))
            ctx.note("bare 'avoid X' treated as forbidden disc near X")
            text = (text[: m.start()] + " " + trailing).strip()
    return text, avoids


def _parse_instruction(text: str, ctx: _Ctx) -> tuple[list[RouteLeg], list[AvoidSpec]]:
    """Segment instruction text into ordered route legs plus whole-traversal avoid specs."""
    text, avoids = _extract_avoids(text, ctx)
    matches = list(_LEG_RE.finditer(text))
    legs: list[RouteLeg] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = _clean_segment(text[m.end() : end])
        if not body:
            continue
        kind = _leg_kind(m.group(1))
        if kind is LegKind.CORRIDOR_BETWEEN:
            corridor_body, extra = _split_continuation(body)
            legs.append(
                RouteLeg(kind=kind, anchors=_split_pair(corridor_body, ctx, what="corridor"))
            )
            if extra:
                legs.append(RouteLeg(kind=LegKind.GOTO, anchors=[_parse_anchor(extra, ctx)]))
        elif kind is LegKind.VIA_NEAR:
            via_body, extra = _split_continuation(body)
            legs.append(RouteLeg(kind=kind, anchors=[_parse_anchor(via_body, ctx)]))
            if extra:
                legs.append(RouteLeg(kind=LegKind.GOTO, anchors=[_parse_anchor(extra, ctx)]))
        else:
            legs.append(RouteLeg(kind=kind, anchors=[_parse_anchor(body, ctx)]))
    if not legs:
        noun = vocab.find_first_noun(text) or "object"
        ctx.note("no route legs matched; single best-effort goto emitted")
        legs = [RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun, raw=noun)])]
    if legs[-1].kind is not LegKind.GOTO:
        if legs[-1].kind is LegKind.VIA_NEAR:
            legs[-1] = RouteLeg(kind=LegKind.GOTO, anchors=legs[-1].anchors)
            ctx.note("terminal via_near leg coerced to goto")
        else:  # terminal corridor: stop just past the gate (second anchor)
            legs.append(
                RouteLeg(kind=LegKind.GOTO, anchors=[copy.deepcopy(legs[-1].anchors[1])])
            )
            ctx.note("terminal corridor leg: appended goto at second corridor anchor")
    return legs, avoids


# --------------------------------------------------------------------- entry point


def parse_regex(question: str) -> Plan:
    """Parse question text into a valid Plan deterministically; total — never raises."""
    try:
        plan = _parse(question)
        if plan.validate():
            plan = _fallback(question, f"structural errors: {plan.validate()}")
    except Exception as exc:  # noqa: BLE001 — the regex tier must be total
        plan = _fallback(question, f"regex tier exception: {exc!r}")
    return plan


def _parse(question: str) -> Plan:
    """One full parse attempt (may produce an invalid Plan; caller re-checks)."""
    qtype = classify_qtype(question)
    ctx = _Ctx()
    text = _normalize(question)
    if qtype is QType.NUMERICAL:
        target = _parse_numerical(text, ctx)
        return Plan(qtype=qtype, question_raw=question, target=target,
                    notes="; ".join(ctx.notes), parse_tier="regex")
    if qtype is QType.OBJECT_REFERENCE:
        target = _parse_object_reference(text, ctx)
        return Plan(qtype=qtype, question_raw=question, target=target,
                    notes="; ".join(ctx.notes), parse_tier="regex")
    route, avoids = _parse_instruction(text, ctx)
    return Plan(qtype=qtype, question_raw=question, route=route, avoid=avoids,
                notes="; ".join(ctx.notes), parse_tier="regex")


def _fallback(question: str, why: str) -> Plan:
    """Minimal always-valid Plan when the structured parse fails."""
    try:
        qtype = classify_qtype(question)
    except Exception:  # noqa: BLE001
        qtype = QType.OBJECT_REFERENCE
    noun = vocab.find_first_noun(question) or "object"
    note = f"fallback plan ({why})"
    if qtype is QType.INSTRUCTION_FOLLOWING:
        return Plan(
            qtype=qtype, question_raw=question,
            route=[RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun, raw=noun)])],
            notes=note, parse_tier="regex",
        )
    return Plan(
        qtype=qtype, question_raw=question,
        target=TargetSpec(noun=noun, raw=noun),
        notes=note, parse_tier="regex",
    )
