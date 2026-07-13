# Diagnosis: GT-battery numerical count disagreements

Judgment pass over docs/resolver-dump.md (instance-level
evidence, generated 2026-07-14 from the re-downloaded VLA-3D
Unity subset). Battery source:
reports/gt_battery_full_2026-07-11/.

Bottom line: the 15%/27% "independent agreement" number was
never a calibration problem. It decomposes into four REAL
resolver defects (code fixes), three scorer-side measurement
artifacts (fix the second opinions, not the resolver), and a
small calibration residue. The planned k-fold CV sweep must not
run until the code fixes land, and must never score against the
class-only or direction-inverted opinions.

## The four resolver defects (code fixes)

### D1. Relaxation ladder leaks into counting()
`counting()` (toolbox.py:545) counts `resolve()` survivors.
`resolve()`'s fallback ladder (relax attributes -> drop
relation -> category-only) is grounding behaviour - "return
SOMETHING for object-reference" - but for a count question it
silently rewrites the question: a relation matching nothing
returns ALL class instances, and `counting()` discards the
audit trail that records this. Fired on 7 of 15 questions
(home_building_1, home_building_2, hotel_room_1, hotel_room_2,
livingroom_4, loft, studio). home_building_1's absurd 31 is
exactly this: 0 survivors -> category_only -> 31. Worse,
hotel_room_1 - the ONE scene the battery reported as clean
agreement (4=4=4) - also got its 4 from category_only: right
answer, wrong mechanism, pure luck.
Fix: counting() must use strict (unrelaxed) resolution, or
consume the audit and flag/zero relaxed counts. Calibration
cannot fix this.

### D2. Noun-matcher pollution and misses (by_label)
Typo tolerance (Levenshtein <= 2) is applied to ALIASES, and
GT aliases are COLOR names. 'yellow' is distance 2 from
'pillow': every yellow-dominant object matches the noun
"pillow". home_building_1's pillow pool contains umbrella,
face cream, bottles, cookers, slipper, dining table, lamps
(13 aliens of 31); livingroom_4 gained a mirror;
livingroom_2 matched 'tap' as "cup"; hotel_room_2 matched
anchor 'bag' as "bed". Meanwhile the matcher MISSES head-noun
compounds: 'coffee cup' does not match "cup" (livingroom_2's
real under-count). Fix: exclude color/attribute aliases from
typo matching, scale typo distance to token length, add
head-noun (suffix-token) matching. Code fix.

### D3. on() z-band semantics fail supporters with backrests
`on()` requires a's bottom within +/-0.15 m of b's AABB TOP.
Sofas and beds carry backrests/headboards: the AABB top is
0.4-1.1 m above the seat/mattress surface, so every genuinely
on-the-sofa pillow FAILS (vgaps -0.36..-0.63 home_building_1,
-1.08 hotel_room_1, -0.40 livingroom_4) - which is what
triggered D1's relaxation on all pillows-on-sofa questions.
Footprint overlap for the true positives is 99-100%, so the
overlap term is healthy; the z-band is the broken term.
Fix: support semantics = footprint overlap AND bottom inside
the supporter's upper z-span (bottom >= zmin, bottom <= ztop +
tol) - "resting on or in". Note this also ADJUDICATES AGAINST
the dossier prior of a 1 cm directional band at AABB top
(sort3d/vla_3d `on_thres=0.01`): the generator's own edge
formula has the same blindness - its sg 'on' has only 198
edges and does NOT contain pillows->sofa - yet the QUESTIONS
still say "on the sofa", so question semantics are broader
than both. Code fix + params sweepable after restructure.

### D4. above()/under() boolean footprint-overlap gate rejects
### wall-hung objects
"Pictures above the bed" (hotel_room_2): all 5 fail on
footprint overlap - wall-mounted pictures do not overlap the
bed footprint at all; GT relation-aware answer is 3. Same for
studio "framed records above the couch". The dossier prior
(add vertical_iom >= 0.2-0.5) goes the WRONG WAY: even
any-overlap is already too strict for the wall-hung case the
questions actually ask about. Fix: replace the boolean gate
with a lateral-offset tolerance (XY centre within an inflated
anchor footprint), tolerance sweepable. Code fix; evidence
overrides the dossier here.

## The three scorer-side artifacts (fix the measurement)

### S1. "referential" second opinion never checks the relation
`_independent_count` rel_ids require only that the ANCHOR
CLASS appears in the statement's anchors - the relation itself
is never compared. livingroom_1's indep=9 includes a chair
7.21 m from the named table (its statement anchors a different
table); livingroom_3's indep=10 counts wall pictures with ANY
tv-cabinet statement as "on" it. The "referential" tag
overstates what was verified. Fix: match the relation phrase
too, or retag as anchor-class-only.

### S2. Scene-graph opinion reads asymmetric relations inverted
VLA-3D stores `relationships[rel][X] = [things REL'd to X]`
(anchor-centric): `on[sofa] = [pillows on it]`. The scorer
checks `edges[rel][target] contains anchor` - i.e. for
"pillows on sofa" it asks "is the sofa on the pillow". All its
relation-aware wins came through the SYMMETRIC 'near' edges
(direction-proof), so sg counts are really NEAR counts.
office_1's sg=12 adds token-overlap pollution ('computer
mouse' matches "computer monitor" via the shared token). The
'between' pair-shape str() bug is confirmed real but latent
(no numerical question routes through 'between').
Fix: invert lookup direction for on/above/below/hanging_on;
tighten `_anchor_agrees`; handle pair-shaped between.

### S3. class_only rows are not evidence of error
Where the winning opinion is `*_class_only` (arabic_room,
japanese_room, office_2, and the sg side of hotel_room_2 /
studio), the comparison is relation-blind by construction.
The pipeline was RIGHT (or unfalsified) in all of them.

## Calibration residue (legit sweep material)

- Question colors vs the 15-color scheme: "red" pillows are
  annotated maroon/purple; "black" pillows (loft) match NO
  scheme color at all. Needs a question-color -> scheme-color
  synonym map (red~maroon; investigate loft's black - possibly
  a color outside the top-3 schemes).
- near threshold form (T4 #1, region-volume vs footprint-diag):
  the only near-driven question (livingroom_1) PASSED with the
  current adaptive form (gap 0.00 vs 1.40 thresh, agrees with
  sg=8). Keep as a sweep dimension, not a proven defect.
- on()/above() tolerances after D3/D4 restructure.

## Per-question verdicts

| Scene | pipeline vs best-GT | Cause bucket | Verdict |
|---|---|---|---|
| arabic_room | 1 vs (3 class-only) | S3 granularity | pipeline right; no fix |
| chinese_room | 6 vs 1(ref)/6(sg) | annotation coverage (1 statement annotated) | pipeline right; no fix |
| home_building_1 | 31 vs ~11-18 | D1+D2+D3 | code |
| home_building_2 | 10 vs 4 | D1+D3 + color map (red~maroon) | code + calib |
| hotel_room_1 | 4 vs 4 (lucky) | D1+D3 masked | code (silent) |
| hotel_room_2 | 5 vs 3 | D1+D2(bag~bed)+D4 | code |
| japanese_room | 3 vs (4 sg class-only) | S3 + sg token match | pipeline right; no fix |
| livingroom_1 | 8 vs 9(ref)/8(sg) | S1 (anchor-specificity) | pipeline right; scorer fix |
| livingroom_2 | 1 vs 2 | D2 (coffee cup missed; tap polluted) | code |
| livingroom_3 | 2 vs 10/9 | S1 + genuine "on vs hung-above" ambiguity | scorer fix; flag ambiguity |
| livingroom_4 | 7 vs 6 | D1+D2(mirror)+D3 | code |
| loft | 11 vs ~8-9 | D1 + color map (black unmapped) | code + calib |
| office_1 | 6 vs 6(ref) | S2 (sg=12 is mouse pollution) | pipeline right; scorer fix |
| office_2 | 1 vs (3 class-only) | S3 granularity | pipeline right; no fix |
| studio | 3 vs (1 sg class-only) | D1+D4 (right count, wrong mechanism) | code (silent) |

Score after separating artifacts: the resolver was genuinely
wrong on ~6 of 15 (all relaxation/matcher/predicate defects),
genuinely right on ~7, and ambiguous on 2 - far from the 15%
agreement headline, but the wrongness is structural, not a
threshold-tuning problem.

## What the k-fold sweep may optimize

- ONLY after D1/D2 land (else it tunes against relaxation
  noise and alias pollution).
- Objective: agreement with (a) direction-corrected sg
  relation-aware counts and (b) relation-checked referential
  counts. NEVER against class_only rows (S3) or the current
  inverted/anchor-only opinions (S1/S2).
- Sweepable: D3 support-band params, D4 lateral tolerance,
  on_min_overlap_frac (centre 0.5 per dossier priors), near
  form (current adaptive vs region-volume 0.01*V), color
  synonym map on/off.

## T4 closures established by this evidence

- T4 #2 (target_index id-space): CLOSED-VERIFIED. All 15
  scenes: every referential target_index exists as an
  object_id (100%). Caveat: 5-25% of statements carry a
  target_class that does not loosely match that object's
  raw_label (e.g. japanese_room 41/54) - the id join is
  sound, the class labels drift.
- T4 #5a (pair shape): RESOLVED, split verdict. `between` IS
  pair-shaped (list of [id,id] pairs) in all 15 scenes -
  research right; `hanging_on` is a FLAT scalar list -
  research wrong on that half. The scorer's uniform flattening
  str()-garbles between pairs (latent; no current question
  hits it).
- Bonus: `beside` exists as a key in all 15 graphs with ZERO
  non-empty edges anywhere; `in` is near-empty (20 edges
  total). Do not rely on either as a GT signal.
