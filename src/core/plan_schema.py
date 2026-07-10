"""Typed plan DSL — the output contract of the checkpoint-1 parse.

Design constraints (docs/question_analysis.md, verified 10 Jul 2026):
- All training relations are allocentric object-to-object; dominant predicates:
  on(48), closest_to(37), near(33), between(17), with(11); plus corridor forms
  `path between` / `path near` and penalty-scored `avoid` in instruction-following.
- 100% of instruction-following questions are multi-constraint and ordered.
- Attributes are rare (10 mentions total) but must round-trip when present.
- Parser must tolerate typos ("refridgerator"); nouns normalise lowercase-singular.

The schema is deliberately CLOSED (enum predicates, no free-form geometry): novel
phrasing degrades to the nearest predicate + a `notes` escape hatch that the
verification checkpoint can inspect — never to silent free-text.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum

from core.interfaces import QType


class Pred(str, Enum):
    ON = "on"  # support/containment-above: a rests on b
    IN = "in"  # containment: a inside b (room or container)
    NEAR = "near"  # proximity, scale-adaptive threshold
    NEXT_TO = "next_to"  # adjacency, tighter than near
    BETWEEN = "between"  # a within capsule(b1, b2)
    ABOVE = "above"  # vertical, no support
    UNDER = "under"
    CLOSEST_TO = "closest_to"  # superlative over candidate set
    FARTHEST_FROM = "farthest_from"
    WITH = "with"  # possession/feature: "the table with the lamp"


@dataclass
class Anchor:
    """A referenced object: noun + optional attributes + optional nested disambiguator.

    noun: canonical lowercase-singular ('fridge', not 'Refridgerators').
    raw: the surface form as it appeared, for typo-tolerant matching + audit.
    """

    noun: str
    raw: str = ""
    attributes: list[str] = field(default_factory=list)  # e.g. ["blue", "small"]
    disambiguator: "Clause | None" = None  # e.g. the fridge NEAR the window


@dataclass
class Clause:
    """One spatial constraint: pred(subject, *anchors). Subject is implicit
    (the target being resolved); anchors are the reference objects."""

    pred: Pred
    anchors: list[Anchor]  # 1 anchor normally; 2 for BETWEEN
    negated: bool = False  # "not near the door"


@dataclass
class TargetSpec:
    """What to count (numerical) or select (object_reference)."""

    noun: str
    raw: str = ""
    attributes: list[str] = field(default_factory=list)
    clauses: list[Clause] = field(default_factory=list)  # ALL must hold (AND)


class LegKind(str, Enum):
    GOTO = "goto"  # terminal or intermediate destination anchor
    VIA_NEAR = "via_near"  # pass near anchor ("take the path near the window")
    CORRIDOR_BETWEEN = "corridor_between"  # thread the gap between two anchors


@dataclass
class RouteLeg:
    """One ordered step of an instruction-following route."""

    kind: LegKind
    anchors: list[Anchor]  # 1 for GOTO/VIA_NEAR, 2 for CORRIDOR_BETWEEN


@dataclass
class AvoidSpec:
    """Penalty region active for the WHOLE traversal (not one leg)."""

    between: list[Anchor] | None = None  # forbidden corridor between two anchors
    near: Anchor | None = None  # forbidden disc around one anchor


@dataclass
class Plan:
    """Checkpoint-1 output. Exactly one of target/route is populated per qtype."""

    qtype: QType
    question_raw: str
    target: TargetSpec | None = None  # NUMERICAL & OBJECT_REFERENCE
    route: list[RouteLeg] = field(default_factory=list)  # INSTRUCTION_FOLLOWING
    avoid: list[AvoidSpec] = field(default_factory=list)
    notes: str = ""  # escape hatch: anything the parser couldn't encode; verification reads this
    parse_tier: str = "api"  # "api" | "api2" | "local" | "regex" — audit provenance

    # ---------------------------------------------------------------- validation

    def validate(self) -> list[str]:
        """Return a list of structural problems; empty list == valid.

        Checks FORM (the semantic check is the verification checkpoint's job).
        """
        errs: list[str] = []
        if self.qtype in (QType.NUMERICAL, QType.OBJECT_REFERENCE):
            if self.target is None:
                errs.append(f"{self.qtype.value} plan requires target")
            if self.route:
                errs.append(f"{self.qtype.value} plan must not carry a route")
        if self.qtype is QType.INSTRUCTION_FOLLOWING:
            if not self.route:
                errs.append("instruction_following plan requires >=1 route leg")
            if self.target is not None:
                errs.append("instruction_following plan must not carry a target")
            if self.route and self.route[-1].kind is not LegKind.GOTO:
                errs.append("route must terminate in a GOTO leg")
        for i, leg in enumerate(self.route):
            want = 2 if leg.kind is LegKind.CORRIDOR_BETWEEN else 1
            if len(leg.anchors) != want:
                errs.append(f"route[{i}] {leg.kind.value} needs {want} anchor(s)")
        if self.target is not None:
            for j, cl in enumerate(self.target.clauses):
                want = 2 if cl.pred is Pred.BETWEEN else 1
                if len(cl.anchors) != want:
                    errs.append(f"target.clauses[{j}] {cl.pred.value} needs {want} anchor(s)")
        for k, av in enumerate(self.avoid):
            if (av.between is None) == (av.near is None):
                errs.append(f"avoid[{k}] must set exactly one of between/near")
            if av.between is not None and len(av.between) != 2:
                errs.append(f"avoid[{k}].between needs exactly 2 anchors")
        return errs

    # ------------------------------------------------------------- serialisation

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=_enum_value, indent=None)

    @staticmethod
    def from_json(s: str) -> "Plan":
        return _plan_from_dict(json.loads(s))


def _enum_value(o):  # json.dumps default hook
    if isinstance(o, Enum):
        return o.value
    raise TypeError(type(o))


def _anchor_from_dict(d: dict) -> Anchor:
    return Anchor(
        noun=d["noun"],
        raw=d.get("raw", ""),
        attributes=list(d.get("attributes", [])),
        disambiguator=_clause_from_dict(d["disambiguator"]) if d.get("disambiguator") else None,
    )


def _clause_from_dict(d: dict) -> Clause:
    return Clause(
        pred=Pred(d["pred"]),
        anchors=[_anchor_from_dict(a) for a in d["anchors"]],
        negated=bool(d.get("negated", False)),
    )


def _plan_from_dict(d: dict) -> Plan:
    tgt = d.get("target")
    return Plan(
        qtype=QType(d["qtype"]),
        question_raw=d.get("question_raw", ""),
        target=TargetSpec(
            noun=tgt["noun"],
            raw=tgt.get("raw", ""),
            attributes=list(tgt.get("attributes", [])),
            clauses=[_clause_from_dict(c) for c in tgt.get("clauses", [])],
        )
        if tgt
        else None,
        route=[
            RouteLeg(kind=LegKind(l["kind"]), anchors=[_anchor_from_dict(a) for a in l["anchors"]])
            for l in d.get("route", [])
        ],
        avoid=[
            AvoidSpec(
                between=[_anchor_from_dict(a) for a in v["between"]] if v.get("between") else None,
                near=_anchor_from_dict(v["near"]) if v.get("near") else None,
            )
            for v in d.get("avoid", [])
        ],
        notes=d.get("notes", ""),
        parse_tier=d.get("parse_tier", "api"),
    )
