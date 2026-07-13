# Verdicts: 2025 dossier priors vs architecture v1.0

Adjudication pass (2026-07-14) over docs/mining-table.md
Table C, informed by the instance-level GT evidence in the
numerical-count diagnosis
(../T5-numerical-count-diagnosis/docs/diagnosis.md). Verdicts
are append-only per the provenance rules in
docs/prior_art/README.md: team docs stay authoritative for
build/process; dossiers are the primary record for 2025 facts.

A general ruling that shaped several verdicts below: where a
dossier prior and OUR OWN instance-level GT evidence conflict,
the evidence wins - the dossiers describe how the generator
and other teams computed relations, but the questions'
EFFECTIVE semantics (what the annotations actually count) are
observable directly, and twice they contradict the priors.

## Conflict verdicts

### C1. near functional form - DEFER (sweep both)
Prior: `euclidean(centers) < 0.01 * region_volume`
(generator formula). Ours: `max(1.2, 0.6*footprint_diag)`
AABB gap. Evidence: the single near-driven battery question
(livingroom_1 chairs-near-table) agreed with the scene graph
under OUR form. Verdict: keep the current form as default,
add the region-volume form as a k-fold sweep alternative,
decide empirically. Not folded into code yet; the sweep gate
is the D1/D2 resolver fixes from the diagnosis.

### C2. on() overlap gate 0.30 vs 0.5 - CHANGE (center 0.5)
Three independent sources converge on 0.5, and the GT
evidence shows true positives sit at 99-100% overlap, so
tightening costs nothing and buys precision. Verdict: move
the default to 0.5, keep sweepable.

### C3. on() vertical band: 1 cm directional - REJECT prior
The generator's own 1 cm band shares our current z-band's
blindness: sofas/beds carry backrests, so the AABB top is
0.4-1.1 m above the seat, and NEITHER band matches what the
questions mean by "on the sofa" (the sg 'on' edge set is
tiny - 198 edges - and omits pillows->sofa, yet the questions
ask it anyway). Verdict: reject both the dossier's 1 cm and
our +/-0.15 m band at AABB top; restructure on() to support
semantics (footprint overlap AND bottom inside the
supporter's upper z-span), params sweepable. Diagnosis D3.

### C4. above()/below() overlap gate - REJECT prior, go the
### OTHER way
Dossiers say add a fraction gate (0.2-0.5). Instance evidence
says even our boolean any-overlap gate is too strict: GT
counts wall-hung pictures "above the bed" whose footprints do
not overlap the bed at all. Verdict: replace the overlap gate
with a lateral-offset tolerance (XY centre within inflated
anchor footprint), sweepable. Diagnosis D4. This is the
clearest case of evidence-over-dossier.

### C5. between form - KEEP (defer redesign)
The sources disagree with each other (between_iom 0.5 vs
0.05), no battery numerical question routes through between,
and our capsule test is simpler. Verdict: keep the capsule;
log the projected-frame scheme as a candidate if
object-reference between-questions fail later. The REAL
between action item is scorer-side: the sg stores between as
[id,id] PAIRS and the scorer garbles them (diagnosis S2).

### C6. big/small basis - CHANGE
Two sources converge on relative per-class largest-face-area
with a 1.2x gap; ours is absolute class-agnostic volume.
Size/color attributes also feed object-reference selection.
Verdict: adopt the relative per-class form in the GT loader's
size token (and mirror in perception attributes later).

### C7. above_gap min vs cap - KEEP
The min-gap is a single team's tuning (KAIST-ISE), not the
generator; our cap is unenforced and harmless. Subsumed by
the C4 redesign. No action beyond C4.

## Cross-cutting additions (not conflicts, but actionable)

- Color synonym map: question colors are NOT the 15-scheme
  vocabulary verbatim ("red" pillows annotated maroon/purple;
  loft's "black" matches no scheme color). Add a
  question-color -> scheme-color map; investigate loft black.
- Frontier-scoring avoid-awareness: the URL-KAIST dossier's
  leak (frontier targets ignore the avoid-filtered map) is
  only partially covered by our costmap invariant - the
  architecture's frontier-scoring box does not explicitly
  consume avoid capsules. Flagged for a direct check when the
  frontier planner lands.

## GT-markers implication (T4 hole #4 fold)

Ruling: fold into docs/architecture.md and
docs/master_plan.md (the one place this was still absent).
Content of the fold:
- 2 of the 2025 top 3 (ReasonX 2nd, CopyPasta 3rd; URL-KAIST
  4th likewise) answered from the sim's ground-truth
  `/object_markers` topic - legal in 2025, absent from the
  2026 I/O list.
- Consequence: 2025 scores are NOT a calibration baseline for
  2026 perception difficulty. The open-vocab perception stack
  faces a categorically harder job than the reported methods
  imply, and perception quality (detector recall + attribute
  extraction) is the single largest differentiator vs teams
  porting 2025 pipelines.
- Also correct architecture.md's stale line ("2025
  third-place configuration is known only from a CMU MRSD
  newsletter item") - superseded by the full dossier set
  under docs/prior_art/.
