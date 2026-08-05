"""Shared established-instance SceneIndex view (#151, generalized to OBJECT_REFERENCE by #184).

Both NUMERICAL's relation-clause anchor filtering (``core.heads.numerical``, issue #151)
and OBJECT_REFERENCE's candidate/anchor ranking (``core.heads.object_ref``, issue #184) hit
the SAME phantom-track failure mode against the SAME toolbox machinery: single/few-
observation same-class detector noise (a "ghost" instance, as opposed to a real, repeatedly
re-observed track) wins a relation clause, a candidate rank, or a superlative anchor pick it
has no business winning, because neither ``core.geometry.toolbox.counting()`` nor
``.resolve()`` apply an observation-count floor of their own — their predicates are
label/geometry-only by design.

Rather than teach the toolbox predicates about observation counts (which would apply the
gate to every caller, including ones that legitimately want the raw pool), or duplicate the
established/ghost distinction per head, each head wraps the ``SceneIndex`` it hands to its
one toolbox call in :class:`EstablishedView`, scoped to exactly that call; the view never
mutates the underlying index.

``by_label``/``by_label_tiered`` are the only ``SceneIndex`` methods either toolbox path
reaches (``counting()`` -> ``_filter_and`` -> ``_eval_clause`` -> ``_resolve_anchor``;
``resolve()`` -> ``_match_noun`` / ``_match_anchor_noun`` / ``_resolve_anchor``), so wrapping
just those two is sufficient to gate every lookup the call makes through the SAME toolbox
machinery each head already uses — no toolbox code is modified or duplicated.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.interfaces import InstanceRecord, MatchTier, SceneIndex

#: An instance is "established" once it has been observed this many times — the single
#: threshold both heads use for "is this a real, tracked instance, or single/few-frame
#: detector noise". Shared by NUMERICAL's target/anchor gating (H15(b)/#151) and
#: OBJECT_REFERENCE's candidate/anchor gating (#184): one number reused across issues
#: (generalization protocol), not refit to either issue's own sample.
ESTABLISH_N_OBS: int = 3


@dataclass(frozen=True)
class EstablishedView:
    """SceneIndex wrapper narrowing ``by_label``/``by_label_tiered`` lookups to instances
    with ``n_obs >= floor``, exempting ``exempt_noun`` (if set), which passes straight
    through untouched.

    ``exempt_noun`` exists for NUMERICAL (#151): its target noun already carries its OWN,
    separate min_obs gate (``NumericalHead._answer_min_obs``, H15(b)) applied by the caller
    before ``counting()`` even runs, so double-gating the SAME noun here would double-count
    observations toward one establishment decision — only the ANCHOR side of a relation
    clause is meant to be gated by this view for NUMERICAL. OBJECT_REFERENCE (#184) has no
    separate target gate of its own, so it leaves ``exempt_noun`` unset: every lookup —
    candidate pool AND anchor pool alike — is gated identically.

    ``all_instances()`` passes straight through: unused by either toolbox path this view
    wraps, kept only so the view remains a total ``SceneIndex`` duck-type for any caller
    that happens to check it (e.g. an ``advance()`` empty-index guard).
    """

    inner: SceneIndex
    floor: int = ESTABLISH_N_OBS
    exempt_noun: str | None = None

    def _passthrough(self, noun: str) -> bool:
        return self.exempt_noun is not None and noun == self.exempt_noun

    def by_label(self, noun: str) -> list[InstanceRecord]:
        recs = list(self.inner.by_label(noun))
        if self._passthrough(noun):
            return recs
        return [r for r in recs if r.n_obs >= self.floor]

    def by_label_tiered(self, noun: str) -> list[tuple[InstanceRecord, MatchTier]]:
        recs = list(self.inner.by_label_tiered(noun))
        if self._passthrough(noun):
            return recs
        return [(r, t) for r, t in recs if r.n_obs >= self.floor]

    def all_instances(self) -> list[InstanceRecord]:
        return self.inner.all_instances()
