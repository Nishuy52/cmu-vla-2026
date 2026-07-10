# Critique from Proposal B (Frontier-VLM Agentic)

**Date:** 10 Jul 2026. Author stance: frontier-VLM agentic system. Rivals: A (deterministic-first
modular), C (expected-score maximisation). Attacks target each rival's *best* version.

---

## 1. Attacks on A

**A1 — The perception stack is a single upstream point of failure, and A admits it can't repair
it.** A concedes this in its own Weaknesses ("Perception is the single point of failure... no
downstream language reasoning can repair it"), but underweights how load-bearing that admission is.
A routes *all three* answer types through GroundingDINO + MobileSAM + a NumPy ByteTrack + lidar
fusion. The training vocabulary has 114 distinct nouns including rare/fine-grained items —
"hookah", "sphere decoration", "framed records", "calligraphy painting" (question_analysis §4). A
closed detector prompt list plus question-noun injection helps, but GroundingDINO's recall on
fine-grained indoor nouns at 90° crop edges is exactly where open-vocab detectors are weakest, and
a systematically-missed category poisons the count, the marker, *and* the instruction-following
anchor simultaneously. A frontier VLM looking at the same panorama degrades far more gracefully on
a novel noun because it isn't gated by a detector's confidence threshold. A has no fallback that
*sees* — its fallback ladder (§6 #6) only relaxes filters over a map that may not contain the
object at all.

**A2 — The frozen single-parse DSL is bet-the-run on the parser, with no in-loop recovery.** A's
central claim is that removing the LLM from the answer path removes stochasticity (§0). But it
relocates *all* language risk into one up-front parse that is never revisited (§3.0: "The parse is
done once; the LLM is not consulted again"). A itself flags "single-sample parse... a rare unlucky
parse is caught only by schema validation, which checks form, not meaning" (Weaknesses). Under
no-retry scoring on a 6-point instruction-following question, one semantically-wrong-but-valid
parse silently loses the whole question, and nothing downstream can notice — the deterministic
resolver will faithfully execute the wrong plan. An agentic loop re-observes and can catch "this
doesn't look like a hookah" mid-run; A structurally cannot.

**A3 — Threshold-predicate brittleness on unseen room scales is a real, correlated risk.** A's
`on`/`near`/`between` are constant-tuned on 15 training scenes (§3.0b, "thresholds... tuned offline
on the 15 training scenes"). `near(a,b) ≤ max(1.2 m, ...)` and `between`'s capsule radius are
fixed numbers. The 3 held-out scenes could be larger corporate/multi-room layouts (only 2 of 15
training scenes are multi-room) where "near a sofa" spans differently. A's scale-adaptive term
mitigates but, as A concedes, "doesn't eliminate this." Borderline geometry is precisely where a
VLM judging "is this near, given the room?" from the image can beat a fixed cutoff — A gives up
that judgment by construction.

---

## 2. Attacks on C

**C1 — The expected-points table is the proposal's foundation and its numbers are unfalsifiable
self-estimates.** C's entire prioritisation instrument (§1.2) is a table of "Points unlocked
(E[Δ], /51)" that C admits are "my expected marginal test-set contribution... estimated from
training-set question structure and published accuracy numbers." No ground-truth answers ship in
the JSON (question_analysis §6: "Ground-truth answers do NOT ship"), so *none* of these deltas can
be validated before submission. The whole cut-order (§7) — what survives, what dies — rides on
point estimates that are educated guesses. If the real IF partial-credit curve is steeper or
flatter than assumed, the pts/day ranking inverts and C has optimised the wrong subsystem with no
way to have known. My design also can't see GT, but I don't stake a *cut-order* on fabricated
per-row point values — I keep every question type in the same loop and let a watchdog allocate
clock, not a spreadsheet allocate features.

**C2 — The MVS explicitly ships with corridor geometry degraded to a midpoint via-point — on the
70.6%-of-points question type.** C's MVS (§7) is "no corridor geometry (corridors degrade to a
midpoint via-point)" and its week-2 cut-line degrades avoid-capsule A* to "leg-wise straight
routing." But question_analysis §7 shows `path between` appears 10× and `avoid(ing)` corridors are
penalty-scored — and C's own §4.1 says avoidance "scopes the whole traversal." A midpoint via-point
does not guarantee the trajectory *threads* the gap (it can approach and leave from the same side),
and straight-line legs can cut straight through a forbidden capsule and eat the penalty. C is
cutting geometry on exactly the type it declares "the business," betting that the MVS ships before
that geometry lands. That's a schedule bet dressed as a scoring bet.

**C3 — C is a hybrid that inherits the API dependence it criticises me for, while adding a second
failure surface.** C leans on the SORT3D-style deterministic toolbox *and* an LLM reasoner making
3–6 tool-calling API calls per question (§5) *and* per-clause verification API calls (§4.2.3) *and*
self-consistency ×3 on IF (§5). That is more API round-trips on the critical path than my design's
answer step in several cases, plus a full local perception stack that must also work. Its
local-7B fallback is, by its own admission, "materially weaker at multi-step parsing, and its
quality gap on IF questions is untested until week 3" (§6.2). So C carries both my API-dependence
risk *and* A's perception-single-point-of-failure risk, and calls the combination a hedge. The
verification pass is genuinely good (see §3), but the claim that C is meaningfully less
API-exposed than me is overstated.

---

## 3. What I would steal

- **C's per-clause verification pass (§4.2.3) — steal outright.** A single LLM call after
  selection — "does object #k satisfy EVERY clause? yes/no per clause, using the toolbox's computed
  predicate values" — directly attacks SORT3D's documented compositional-slip failure
  (prior_art §1, organizer_playbook A.1 Fig 3.4). My proposal has a "verify-every-clause" pass
  (§3.2) but C's framing — enumerate clauses, demote to runner-up on any "no" — is sharper. In a
  VLM-agentic frame this is one extra agent step before MARK/ANSWER, feeding the model my printed
  predicate values as evidence; cheap and high-yield on the 2- and 6-point types.

- **C's asymmetric early-answer rule made explicit per type (§4.0).** My watchdog is time-based; C
  adds a *margin*-based stop ("distance-ratio ≥1.5 for closest/farthest; unique candidate
  otherwise; and 45 s of observation didn't change the answer") with an asymmetry — answer
  aggressively on 1-pt numerical, never cut IF exploration to bank a bonus if a sub-goal is
  ungrounded. This is a better trigger than my confidence-field heuristic and folds cleanly into my
  loop as the DONE condition. Integrating: compute the margin from my sighting-ledger coordinates
  (which I already print for the VLM) rather than from a toolbox.

- **A's avoid-corridor-as-planning-invariant (§3.1 step 2, C echoes in §4.1).** Both rivals stamp
  the forbidden capsule into the costmap so A* *cannot* route through it, rather than trusting the
  reasoner to avoid it. My §3.1 already does code-enforced capsules + A* re-route, but A's framing
  — "avoidance is a planning invariant, not a behavior" — is the correct mental model and I should
  state it that way. This is the one place where taking geometry away from the VLM is unambiguously
  right, and I keep it.

- **C's "the floor is not zero" scaffolding discipline (§1.2, §4.5).** The explicit rule that every
  question type always publishes *something* legal (best marker, modal integer, drive at best
  anchor) so no question scores 0 for plumbing reasons. My honest-floor fallback (§4) does this for
  the outage case; C generalises it to *every* run regardless of connectivity. Worth adopting as a
  first-class invariant, not just a degraded-mode behavior.

---

## 4. Non-negotiables (must survive any merge)

- **NN1 — VLM-in-the-loop re-observation, not a frozen up-front plan.** The single element that
  distinguishes B from both rivals: the reasoner is consulted *every step* against fresh imagery,
  so a wrong grounding, a novel noun, or a mis-parse can be caught and corrected mid-run. A freezes
  its plan after one parse and cannot recover (attack A2); C re-grounds but still leans on a
  detector that can silently miss the object. On the 114-noun open-vocabulary distribution
  (question_analysis §4) and 3 unseen test scenes, the ability to *look again and reconsider* is
  the highest-value property in the whole design and must survive.

- **NN2 — Panorama fed whole to the reasoner for exploration decisions.** Both rivals split the
  1920×640 strip into 4 crops and throw the full-ring context away before any decision is made —
  fine for their detector-first perception, fatal for agentic look-around, which needs to see all
  360° at once to choose a direction (VLFM-style, arXiv:2312.03275). Tiling stays only for MARK
  refinement and counting crops (my §5.2). The whole-panorama exploration decision is what makes
  the frontier VLM better-informed than an embedding value map, and it is core to the bet.

- **NN3 — Lidar-frustum back-projection for the Marker box.** Extents come from `/registered_scan`
  points inside the VLM's chosen angular frustum, never from the image — because the Marker is
  scored by GT-box overlap and a VLM has no metric sense (upstream_notes §3, "scored by overlap...
  center is also used as a navigation waypoint"; organizer_playbook A.2, wide-FOV visual geometry
  fails to generalise). This is the honest concession that keeps geometry deterministic exactly
  where B is otherwise weakest, and removing it would make my worst-scored output (wall-mounted /
  thin objects) even worse.

---

## 5. Concessions

- **The API-dependence attack lands, and it is my single biggest exposure.** Both rivals ship a
  local-model fallback baked into the image (A: Qwen2.5-7B for parsing only, needed on *one* call;
  C: Qwen2.5-7B reasoner auto-switch). A is genuinely more outage-robust than me: its *answer path
  needs no network at all* (A §5) because the LLM touches only the up-front parse. If the eval
  network is dark for a full 10 minutes, A scores near-normal and I score near-floor. My
  dual-provider failover + offline geometric mode + honest-floor answers (§4) reduce but cannot
  erase this — it is structural to the bet. Honest mitigation direction: adopt a local-VLM
  degraded tier (a quantised 7B multimodal model in the image) so my *offline* mode still reasons
  over images rather than falling back to pure geometry — narrowing, not closing, the gap to A.

- **Geometric imprecision on the Marker is real and rivals beat me on it.** Both build a proper
  tracked-instance object map with cross-frame fusion and percentile-trimmed extent fitting (A §3.2,
  C §4.2.4). My single-frustum lidar cluster fit is cruder and will lose IoU on partially-occluded,
  thin, or wall-mounted objects (wall lamps, photos, framed records — all in the training
  vocabulary). This is 23.5% of points where my extents are second-best by construction. Partial
  mitigation: accumulate frustum points across the 2–3 approach frames I already capture before
  MARK, and trim outliers as they do — closing part of the gap without abandoning the VLM-picks /
  lidar-measures split.

- **Stochasticity compounds across my many decisions in a way theirs does not.** A's answer path is
  deterministic; C samples the LLM only a few times per question. My loop samples a reasoner across
  15–35 steps, so run-to-run variance (SORT3D reports ±6% per grounding call, prior_art §1) stacks.
  Temperature-0 actions and watchdog convergence bound the tail, and multiple-submission /
  highest-counts helps at the meta level, but two runs of my identical code will diverge more than
  either rival's. This is an accepted cost of NN1, not a bug I can fully remove.

- **C's scoring-first discipline exposes that my proposal under-specifies cut-lines.** C ends every
  week submittable with an explicit MVS and cut-order; my plan (§7) is milestone-based but doesn't
  name what gets dropped first if the Ubuntu reinstall slips past Jul 28. That's a real planning
  gap I should close by defining a B-flavoured MVS (loop + whole-panorama exploration + marker
  back-projection + watchdog floor; verification pass and counting ledger as cuttable).

---

### Sharpest single attack per rival
- **On A:** the frozen single-parse DSL bets a 6-point question on one never-revisited parse with
  no in-loop recovery (A2) — one valid-but-wrong parse silently loses the whole question.
- **On C:** the expected-points table that drives the entire cut-order is unfalsifiable
  self-estimated deltas over answers that don't ship (C1) — the prioritisation could be optimising
  the wrong subsystem with no way to know.

### #1 non-negotiable
**NN1 — VLM-in-the-loop re-observation.** The ability to look again and reconsider mid-run is B's
whole reason to exist and the one property neither rival has; everything else in B is negotiable
around it.
