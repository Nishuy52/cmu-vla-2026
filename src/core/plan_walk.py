"""Shared recursive walk over ``Plan`` anchor structures (issue #95).

``Anchor.disambiguator`` (``core.plan_schema``) is itself a full ``Clause`` — e.g. "the
potted plant closest to the pyramid candle holder" nests the "pyramid candle holder"
anchor inside the "potted plant" anchor's disambiguator. That nesting is unbounded in
the schema: a disambiguator's clause carries anchors that can themselves carry a
disambiguator, and so on.

Multiple sites across the codebase walk a plan's anchors (to build a detector prompt's
noun vocabulary, to check whether a target is groundable, to build ground-truth scoring
sets, ...) and historically only looked at the top level, so a noun that appears ONLY
as a nested disambiguator was invisible to them — most costly at the detector-prompt
site, which then never even asks the object detector to look for that class.

This module is the one place that knows how to descend through a disambiguator chain,
so every call site sees the same recursive structure and applies its own projection
(a noun, a normalized label, "first one wins", "all of them") on top.
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator

#: Hard depth cap protecting against a malformed/adversarial plan with a pathologically
#: deep (or, via a hand-built/shared object graph, cyclic) disambiguator chain. No real
#: spoken instruction nests disambiguators more than one or two deep; this bound exists
#: purely so a bad plan degrades to "stops early" instead of hanging or blowing the stack.
MAX_DISAMBIGUATOR_DEPTH = 8


def iter_anchor_chain(
    anchor: Any, *, max_depth: int = MAX_DISAMBIGUATOR_DEPTH
) -> Iterator[Any]:
    """Yield ``anchor`` then, depth-first, every anchor reachable through nested
    ``anchor.disambiguator`` clauses, outer-to-inner, left-to-right.

    Order is well-defined so callers that want "the first noun" can just take the
    first item, and callers that want "every noun" can consume the whole iterator.

    Tolerant like the rest of the plan-walking call sites: uses ``getattr`` so it
    accepts both real ``core.plan_schema.Anchor`` instances and duck-typed
    stand-ins used in tests/mocks, and a ``None`` anchor simply yields nothing.

    Bounded against both excessive depth and cycles: ``max_depth`` caps how many
    disambiguator hops are followed, and an id-based ``seen`` set (shared across the
    whole call, not per-branch) stops a cyclic object graph from looping forever even
    if it somehow got past construction.
    """
    seen: set[int] = set()
    yield from _walk(anchor, max_depth, 0, seen)


def _walk(anchor: Any, max_depth: int, depth: int, seen: set[int]) -> Iterator[Any]:
    if anchor is None or depth > max_depth:
        return
    key = id(anchor)
    if key in seen:
        return
    seen.add(key)
    yield anchor
    clause = getattr(anchor, "disambiguator", None)
    if clause is None:
        return
    for sub in getattr(clause, "anchors", None) or []:
        yield from _walk(sub, max_depth, depth + 1, seen)


def iter_clause_anchors(clauses: Iterable[Any] | None) -> Iterator[Any]:
    """Anchors of ``clauses`` (e.g. ``TargetSpec.clauses``), recursively through each
    anchor's disambiguator chain, in clause order then anchor order."""
    for clause in clauses or []:
        for anchor in getattr(clause, "anchors", None) or []:
            yield from iter_anchor_chain(anchor)


def iter_target_anchors(target: Any) -> Iterator[Any]:
    """Anchors referenced by a ``TargetSpec`` (its clauses only — NOT ``target.noun``
    itself, which is not an ``Anchor``), recursively through disambiguators."""
    if target is None:
        return
    yield from iter_clause_anchors(getattr(target, "clauses", None))


def iter_route_anchors(route: Iterable[Any] | None) -> Iterator[Any]:
    """Anchors referenced by a plan's route legs, in leg order then anchor order,
    recursively through disambiguators."""
    for leg in route or []:
        for anchor in getattr(leg, "anchors", None) or []:
            yield from iter_anchor_chain(anchor)


def iter_avoid_anchors(avoid: Iterable[Any] | None) -> Iterator[Any]:
    """Anchors referenced by a plan's ``AvoidSpec`` list (``between`` and ``near``),
    recursively through disambiguators."""
    for spec in avoid or []:
        for anchor in getattr(spec, "between", None) or []:
            yield from iter_anchor_chain(anchor)
        near = getattr(spec, "near", None)
        if near is not None:
            yield from iter_anchor_chain(near)
