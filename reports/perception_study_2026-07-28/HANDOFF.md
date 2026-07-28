# Perception lane — handoff, 28 Jul 2026 (late)

Companion to `STUDY.md` (the measurements) — this is the operational state.

## Queued and running without supervision

Three chained cluster batches, submitted against `main` at `a683328` (the mirror
`~/vla/src` was rsynced at submit, so they run the full merged stack):

| job | type | questions | depends on |
|---|---|---|---|
| 701982 | numerical | 15 | 701540 |
| 701983 | numerical (repeat) | 15 | 701982 |
| 701984 | object_reference | 30 | 701983 |

Harvest each with `tools/cluster/live_run/harvest_verify.sh <job>`.

**The repeat is deliberate.** Only ONE numerical sweep has ever been recorded
(701118 = 1/15), so numerical has no error bar. #140 measured IF's noise floor at
~0.04 from two identical sweeps; 701982 vs 701983 gives the numerical equivalent.
Without it a 1/15 -> 3/15 result is uninterpretable.

**No IF batch was queued** — a sweep costs ~175 min for a number #140 showed we
cannot resolve below ~0.08.

### Predictions, recorded before the results land

Everything merged after `728cc50` moved office_1 replay by 134 -> 133 instances and
mean IoU 0.107 -> 0.110. So: numerical should stay near 1-2/15, object_reference
should stay near 0.000 (the oracle bound is 0.060, #142). **If either moves
materially, that is unexplained and worth chasing rather than celebrating.**

## Do not retry these — refuted by measurement

| # | hypothesis | verdict |
|---|---|---|
| #131 | cross-tile NMS fixes over-proposal | tiles overlap **0 deg**; pathological cases unmoved at ANY IoU threshold (office_1 `window` 18->18 at 0.75/0.55/0.35) |
| #130 | extent veto blocks merges (vicious cycle) | veto is **0.7%** of rejections. Latent though — it absorbs 94% of pairs a widened gate admits |
| #128 | widening the association gate merges fragments | recall 17.0->15.2%, duplication 5.58->6.18, IoU 0.107->0.093. Gates are in SERIES |
| #138 | depth-completion of the shell-biased centroid | bias confirmed (+0.19 m, 86-92% of detections) but gated correction = wash, ungated = regression on both scenes |
| — | prior extents replacing observed extents | regression on both scenes |
| #121 | camera-derived colour recovers GT names | **unvalidated, not refuted** — 0 of 3 non-majority colours across two scenes; blocked by recall, see below |

## Colour (#121) — RESOLVED as unvalidated, not refuted

The japanese_room replay finally ran. Result: colour **cannot be validated at
current recall**, and has never once correctly named a non-majority colour.

| scene | GT matched w/ colour | answered | correct | non-majority matched | correct |
|---|---|---|---|---|---|
| office_1 (64% gray) | 15 | 12 | 9 | 2 | **0** |
| japanese_room (32% brown) | **3** | **0** (all abstained) | 0 | 1 | **0** |

Pooled **0 of 3** non-majority objects named. office_1's 75%-of-answered loses to
the trivial "always answer the majority colour" predictor (87% on the same set).

japanese_room matched only 3 of its 63 GT objects (37 instances from 132 keyframes) —
a **recall** failure, not a colour failure. Colour's validation is therefore
downstream of the coverage problem, and cannot proceed until instance matching
improves.

**Verdict: keep the code (it abstains rather than guessing, battery unmoved, the 13
non-colour questions unaffected) but do NOT count it as fixing the two colour
questions.** The loft one is probably unfixable via pixel colour at all: GT's `black`
is literally RGB (0,0,0) with zero variance across 828 slots — an annotation
convention, not an observable.

Re-validate with the majority baseline AND the non-majority slice — raw accuracy is
not a signal, since a majority prior scores well while answering zero colour
questions correctly.

**Do NOT bridge `black`<->`gray` to make loft pass.** `vocab.py`'s COLOUR_BRIDGE
comment explains why that was rejected; it would mis-answer "black X" on every grey
object. User directive this session: correctness fixes, not cheap metric movement.

## Merge hygiene — history is currently unreliable

An unattended parallel session merged several lanes while they were being measured:
`#131` landed despite its own commit message reading "NOT a fix ... not a merge
candidate"; `#134` was built on that refuted premise and merged as two
non-identical commits; `#112` and `#129` were each merged twice. Measurement says
none of it regressed anything (see the 133-instance / 0.110-IoU check above), but
**do not attribute any future sweep delta to a specific commit without re-measuring
on the replay harness.**

That session cannot be messaged (`send_message` refuses: unattended). Structural
isolation via worktrees prevented collisions; it cannot prevent redundancy.

## Unmerged branches worth keeping

- `fix/128-assoc-gate` — the gate change is refuted, but its **`associate()`
  rejection instrumentation** (`REJECT_LABEL`/`REJECT_DISTANCE`/`REJECT_EXTENT`) is
  independently valuable and is what refuted #130.
- `fix/box-geometry` — position-correction and prior-extents behind flags defaulting
  to `False`, with the measured regression recorded in the config comments so nobody
  re-derives it.
- `exp/fusion-lateral-clustering`, `feat/perception-replay-harness` — superseded,
  their content is on `main`.

## Where to work next

`#142`: object_reference is capped at **0.060** even with perfect selection, so it is
a readout of index quality, not a lane. **Numerical is the tractable target** —
counting needs instance *identity*, not box *accuracy*, which is a strictly lower
bar than IoU. From the 701118 breakdown: 9 undercount / 5 overcount / 1 exact, with
4 confident zeros. The undercounts are dominated by flat, wall-mounted objects
(pictures above a bed, photos on a cabinet, framed records) where the frustum lift
has no depth separation to cluster on.
