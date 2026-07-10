# Critique A — Deterministic-First Author's Debate Filing

**Date:** 10 Jul 2026
**Author of:** Proposal A (Deterministic-First Modular Pipeline)

---

## 1. Attacks on B (Frontier-VLM Agentic)

**B1 — The core actuation loop is latency-bound to a degree that breaks its own step budget under
realistic variance.** B's §4 concedes "effective step time ≈ 8–12 s/step" and "~35–55 agent steps"
after reserving for travel, but its own worst-case row is a *20 s API round trip* and a *15 s
stall*. With 15–35 sequential VLM decisions per run, a modest p95 tail (one retry in five steps)
compounds: 20 steps × avg 12 s + 5 retries × 15 s = 315 s of the 540 s window spent *waiting*,
before any travel. B is not step-starved on a good day; it is variance-starved on a normal one, and
the challenge scores a *single* run, not an average. A deterministic frontier planner samples at the
5 Hz terrain-map rate (`upstream_notes.md` §3) with zero network in the loop — the same exploration
that costs B an API call per decision costs A a NumPy A* replan.

**B2 — Putting the VLM *in* the exploration loop contradicts the one lesson B itself imports.**
B's §2 replaces VLFM's trained value map with "ask the VLM which numbered direction leads toward the
target," and its §3.1 hands the VLM a printed coordinate ledger for superlatives — good. But its
*movement* decisions (MOVE(k)) are still VLM judgments made 15–35 times, each a fresh chance for the
exact raw-spatial-reasoning failure the organizers document (`organizer_playbook.md` A.1: LLM given
coordinates picks min-x for "left of"; `prior_art.md` §1). B confines that risk at *answer* time
(printed distances) but re-admits it at *navigation* time. Every MOVE is an unverified spatial call.

**B3 — Runtime API dependence is existential and B says so, but under-weights it against the eval
setup.** B's own weaknesses section admits "if the eval network is down for the full 10 minutes,
this design scores near-floor." The evidence is that the eval machine is a Docker container on a
Simply NUC / i9 (`upstream_notes.md` §5; `challenge_brief.md`), with no guarantee of egress. B's
dual-provider failover (§4) is the right hedge but doubles the external-dependency surface — now
*two* providers' uptime, latency, and rate limits gate the score — and its "offline degraded mode"
(largest-gap heuristic) is precisely A's *primary* exploration policy, admitted as a fallback. When
your fallback is a competitor's baseline, the baseline deserved to be the primary.

**B4 — Marker tightness is B's structurally weakest scored output, and it is 23.5% of the points.**
B §6 concedes lidar-frustum cluster fitting loses to "purpose-built detector + instance segmentation
+ tracked 3-D fusion... especially for partially occluded or thin objects (wall lamps, photos)."
The training set is full of exactly these (`question_analysis.md` §4 noun list: wall lamp, photos,
framed records, pictures, decals). Object-reference is IoU-scored (`upstream_notes.md` §3), so
B forfeits tightness on the type where tightness is the whole score — to save engineering it could
have spent, since the perception stack that fixes this is A's baseline.

---

## 2. Attacks on C (Expected-Score Maximisation)

**C1 — The pts/day table is a spreadsheet of unfalsifiable priors driving irreversible cut
decisions.** C §1.2's "points unlocked (E[Δ])" column is explicitly "my expected marginal
contribution... estimated from training-set structure and published accuracy numbers." Those E[Δ]
values are guesses with no error bars, yet they *rank the cut order* (§7) that decides what ships.
Row 4 (Sequential IF executor) is scored 8–10 pts and row 3 (toolbox) 12–16 — but IF is 70.6% of
points (`question_analysis.md` §2) and *every* IF question is 100% multi-constraint with corridor
and avoid constraints (§7). If the E[Δ] on corridor geometry (cut 8th) is off by a factor of two,
C has optimized its schedule against a mismeasured objective. A does not need the estimates to be
right because it builds the corridor/avoid primitives as non-optional geometry.

**C2 — "Cut corridor geometry to a midpoint via-point" quietly forfeits the highest-value
constraint class.** C §7 lists corridor geometry as the *second-to-last* thing built and its MVS
(§7) explicitly "degrades corridors to a midpoint via-point." But `question_analysis.md` §3 counts
`path between` 10× and `near` 28× in instruction-following, and §7 shows corridors and avoid-regions
are *scored with penalties*, not partial-credit throwaways. A bare midpoint via-point does not
demonstrate the trajectory *threading the gap* (the scored behavior) and does nothing for the 3
`avoid` cases where crossing the forbidden capsule is *penalized*. C's own MVS therefore ships a
system that structurally cannot earn the corridor/avoid sub-scores on ~1/3 of the 70.6%-weighted
type — while C's §1.1 argues IF is "the safest investment." The schedule contradicts the thesis.

**C3 — The MVS-first, "every week submittable" strategy is sound but C conflates *a* score with a
*good* score.** C §0 is right that multiple submissions favor early shipping. But its week-3 MVS
(§7) omits the verification pass, corridor geometry, and counting logic — i.e. it ships the parts
that are *reliable* and defers the parts that earn IF and OR *points*. "≈55–65% of the full design's
score" (C §7) shipped early is good insurance, but the framing rewards banking a floor over closing
the gap on the 70.6% type, and C's own cut-order puts the two highest-leverage IF primitives
(corridor, verification) among the *first* to be sacrificed under schedule pressure. The
expected-value logic is locally correct and globally miscalibrated toward safety.

**C4 — C is A with a schedule bolted on, and inherits every deterministic-first weakness without
adding a hedge.** C §2.2's division of labor ("camera semantics-only, lidar geometry-only, LLM never
computes geometry, deterministic toolbox") is *identical* to A's §0. C's originality (§8) is entirely
the scheduling/expected-value layer, not the architecture. That means C shares A's genuine
weaknesses (closed-predicate brittleness, threshold sensitivity, perception as single point of
failure — A §6 Weaknesses) and adds *no* new robustness for the 3 held-out scenes; C §6.2 admits
this ("deliberately underinvests in robustness for the 3 unseen scenes"). Against A, C offers a
better *plan* but not a better *system* — and a plan that cuts corridor geometry is a worse system.

---

## 3. What I would steal

- **C's expected-points prioritisation and MVS gate (§1.2, §7), stripped of the bad cut-order.**
  A's §7 has a week-by-week plan but no explicit "every week ends submittable" invariant and no
  named minimum viable submission. Adopting C's MVS discipline — a scoring system on the board by
  early August, then ratchet — is pure upside given multiple submissions are allowed
  (`challenge_brief.md`). Integrates directly: A already builds bottom-up (predicates → resolver →
  perception → adapter); I would pin an explicit MVS = {latch + object map + near/on/closest/between
  toolbox + frontier + watchdog} and reorder A's weeks 3–4 so it ships by ~Aug 3, with the fix that
  **corridor/avoid geometry stays inside the MVS**, not cut from it.
- **C's asymmetric early-answer rule (§4.0), which is sharper than A's uniform confidence gate.**
  A §4 gates on "winner beats runner-up by ≥25% AND ≥3 observations." C's per-type asymmetry —
  answer numerical aggressively (upside capped at 1 pt), *never* cut IF exploration to bank a bonus
  if a sub-goal anchor is ungrounded — is a strictly better policy because it prices the bonus
  against the type's downside. Drops straight into A's deterministic FSM as a per-type threshold
  table.
- **C's explicit "answer floor is not zero" scaffolding (§1.2, §4.5).** A has a fallback ladder
  (§6 #6) and hard gates (§4) but never states the invariant as bluntly as C: *always publish a
  marker, always publish an integer, always drive at the best first anchor.* Making "silence is the
  only unforgivable failure" (C §4.5) a first-class watchdog requirement tightens A's own gates.
- **B's dual-provider failover for the *parse* call only (B §4).** A uses the LLM once, for parsing,
  with a local-7B and regex fallback. Adding a second API provider *ahead of* the local fallback
  costs nothing (the parse is one call) and raises parse reliability before degrading to the coarse
  regex parser. This is a strict improvement to A's single-call design with none of B's in-loop API
  exposure.

---

## 4. Non-negotiables (must survive any merge)

- **The LLM is out of the answer loop entirely — one up-front parse, deterministic execution
  after.** This is A's defining bet and the single most evidence-backed decision in the field.
  The organizers' own thesis documents LLM raw-coordinate spatial reasoning failing
  (`organizer_playbook.md` A.1; `prior_art.md` §1), and SORT3D's *fix* is a deterministic toolbox
  the LLM *calls*. A goes one step further and removes even answer-time orchestration, which
  eliminates the ±6% run-to-run LLM variance both rivals concede (B §6, C §6.1) from the graded
  path. Every counted integer, every distance comparison, every corridor gate is verifiable code
  with an offline unit test. This is the property that makes the 70.6%-weighted IF type debuggable
  under no-retry scoring; it cannot be traded for B's agent loop.

- **Avoid-corridor as a hard costmap invariant, and corridor gates as mandatory via-points
  (A §3.1).** Avoidance is penalty-scored (`question_analysis.md` §7) and appears in the IF type
  that is 70.6% of points. A stamps the forbidden capsule OBSTACLE into a per-question costmap so
  A* *physically cannot* route through it — avoidance is a planning invariant, not a behavior to be
  sampled (B) or cut to a via-point (C). This is a few dozen lines of geometry with an outsized,
  irreversible score consequence; it must be in the minimum viable build, not deferred.

- **Geometry from lidar only; camera is semantics-only; counts are set-cardinality over tracked
  instances (A §3.3, §5).** Kachana's thesis directly warns wide-FOV visual geometry fails to
  generalize (`organizer_playbook.md` A.2; `prior_art.md` §5), and OpenEQA shows VLMs count poorly
  (`prior_art.md` §3). A's trimmed-AABB point-cloud boxes beat B's frustum-cluster fits on the
  IoU-scored OR type, and A's tracked-instance NMS counting beats VLM enumeration. This division of
  labor is the perception backbone all three types share; it is not negotiable down to camera-derived
  geometry.

---

## 5. Concessions

- **A has no scheduling discipline as sharp as C's, and that is a real gap.** A §7 is a reasonable
  week plan but lacks C's named MVS, its "every week submittable" invariant, and its explicit
  cut-order. Given multiple submissions with highest-score-kept (`challenge_brief.md`), C's
  bank-early-ratchet framing is genuinely superior risk management, and A should adopt it (§3 above).

- **A's uniform early-answer gate is coarser than it should be.** C §4.0 correctly prices the
  early-finish bonus asymmetrically by type; A's single "25% margin + 3 observations" rule
  under-exploits the fact that a numerical bonus is worth risking almost nothing to grab while an IF
  bonus must never come at the cost of a dropped sub-goal. C exposed a real dullness in A's policy.

- **A's closed-DSL brittleness is a genuine, shared liability that B partially hedges and A does
  not.** If the 3 held-out scenes introduce semantics outside A's predicate set (egocentric phrasing,
  novel relations, pragmatic implicature — `question_analysis.md` §5 notes egocentric is zero in
  *training* only), A degrades to approximations with no reasoning fallback, whereas B's in-loop VLM
  could plausibly wing such cases. A already lists this (A §6 Weaknesses), but B's flexibility on
  genuinely out-of-distribution phrasing is a real advantage A cannot fully answer — only bet
  against on distributional grounds (questions are VLA-3D-generated, `prior_art.md` §1).

- **A's exploration has no semantic value model; on a large multi-room held-out scene this could
  cost budget.** Both B (VLM-scored frontiers) and C (detector-affinity semantic bonus) steer
  exploration toward the question's nouns more aggressively than A's weak `category_bias` term.
  A's assumption that single-room coverage is cheap (`upstream_notes.md` §6: 13/15 training scenes
  single-room) holds for training but is a distributional bet on the held-out three. C's
  detector-affinity frontier bonus is a cheap, deterministic-compatible improvement A should fold in.

---

### Debate summary

- **Strongest attack on B:** the actuation loop is variance-bound — 15–35 sequential VLM decisions
  at 8–12 s each with a 20 s retry tail can consume the majority of the 540 s window *waiting on the
  network*, on a single scored run, where A's deterministic planner samples at the 5 Hz terrain rate
  with zero network in the loop (B1).
- **Strongest attack on C:** its cut-order sacrifices corridor and avoid geometry — the
  penalty-scored primitives of the 70.6%-weighted IF type — among the *first* things dropped, so
  C's own MVS structurally contradicts C's own thesis that IF is "the safest investment" (C2).
- **#1 non-negotiable:** the LLM stays out of the answer loop entirely — a single up-front parse,
  then fully deterministic, unit-testable execution of every count, distance, and corridor gate.
