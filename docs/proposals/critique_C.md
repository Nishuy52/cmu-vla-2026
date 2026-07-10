# Critique C — Proposal C's read of A (deterministic-first) and B (frontier-VLM agentic)

**Date:** 10 Jul 2026 · Author stance: expected-score maximisation. Sources cited inline.

---

## 1. Attacks on A (deterministic-first modular)

**A1 — The single-parse DSL bets the highest-value points on the exact thing that can't be
regression-tested: meaning, not form.** A freezes the query DSL in week 1 and consults the LLM
*once*, then never again (proposal_A §0, §3.0). Its own weakness section concedes schema
validation "checks form, not meaning" (§6, last bullet) and that pragmatic implicature ("the
rightmost pillow" → *on the bed*) is "unhandled by construction" (§6). But 100% of
instruction-following questions are multi-constraint, avg 3.6 relations, up to 33 words
(question_analysis §5) — precisely where a single greedy parse with no answer-time reasoning slips
a clause. SORT3D's *own* documented failure mode is compositional slip (prior_art §1; organizer_
playbook TL;DR #2), and A removes the one mechanism — chain-of-thought re-checking at resolution
time — that catches it, keeping only a *deterministic* verify pass (§3.2) that can only re-check
clauses the parser already extracted correctly. If the parse dropped a clause, the verifier never
sees it. That is a systematic, not random, loss on 70.6% of the points.

**A2 — Threshold predicates are calibrated on training-scene GT object lists and will drift on
3 unseen scenes.** A tunes `on`/`near`/`between` constants offline against the 15 training scenes'
`object_list.txt` (§3.0b, §7 week 2) and admits held-out room scale may miscalibrate `near` (§6
"Threshold sensitivity"). This is the training-overfit risk we all share, but A is *most* exposed:
its entire answer path is fixed thresholds with no fallback reasoning, so a mis-scaled `near` in a
large held-out office produces a confident wrong answer with nothing to catch it. B's VLM and C's
LLM-tool-calling both retain a judgment layer over borderline cases; A deliberately does not.

**A3 — Over-engineering relative to points: the avoid-corridor-as-costmap-invariant and
corridor-gate soft-cost machinery (§3.1 steps 2–4) is real geometry work for ~3–13 training
questions.** `avoid` appears 3× in the whole training set (question_analysis §5); A builds a
per-question costmap copy `C'`, capsule stamping, and a stepwise-shrink recovery when the block
makes the goal unreachable (§6 #9). That recovery path — shrinking the forbidden capsule 20% per
step until A* finds a route — *reintroduces the very violation it was built to prevent*, on the
exact questions where the penalty is the point. Points-per-effort: this is high-effort insurance
that self-defeats in its failure branch. C's whole-traversal capsule inflation (proposal_C §4.1)
is the same idea without the self-cancelling recovery, and C schedules it as an early cut (§7).
A treats it as core.

**A4 — "Perception is the single point of failure upstream of everything" is A's own admission
(§6), and A has no semantic exploration to compensate.** A's frontier scoring is geometry + a weak
detector-only `category_bias` (§2.2), explicitly *not* VLFM (§8 note 3). In a large held-out
multi-room scene, A concedes it "may spend more of the budget exploring than a semantically-guided
policy would" (§6). Under a 10-min clock where the early-finish bonus is real, an exploration
policy that is blind to "kitchens are that way" is leaving both time and points on the table
exactly when the scene is hard — and A cut the one mitigation (semantic frontier value) that B and
C both keep.

---

## 2. Attacks on B (frontier-VLM agentic)

**B1 — Runtime API dependence is an existential, un-hedgeable failure mode on the highest-value
questions, and B says so itself.** "If the eval network is down for the full 10 minutes, this
design scores near-floor... not hedgeable from inside the bet" (proposal_B §6, Weaknesses #1).
The challenge runs on a Simply-NUC eval machine via Docker (upstream_notes §5) with no guarantee of
outbound connectivity; the organizers' own thesis flags cloud-LLM dependence as *the* named
deployment risk (organizer_playbook A.1, "Stated limitations"). B's dual-provider failover
(§4) helps only if the failure is provider-side; a firewalled or offline eval host darkens *both*
providers simultaneously, collapsing B to geometric largest-gap wandering with no grounding — a
near-zero on all 6 instruction-following questions at once. C keeps a local Qwen-7B fallback baked
in the image (proposal_C §5) precisely so the answer path survives a dark network; B cannot, because
the *intelligence itself* is remote.

**B2 — 8–12 s per agent step compounds LLM variance across 15–35 sequential decisions on the
6-point questions.** B's own latency model gives ~8–12 s effective/step (§4) and 35–55 steps
available, "not step-starved... variance-exposed" (§4). SORT3D reports ±6% run-to-run variance for
a *single* LLM grounding call (prior_art §1); B chains 15–30 such calls per instruction-following
question, each a branch point where a bad sample sends the robot down the wrong doorway and burns
travel time that can't be recovered. B's mitigation is temperature-0 + watchdog (§6 #2), but
temperature-0 doesn't make a wrong-but-confident MOVE right, and the watchdog only guarantees *an*
answer, not a good one. This is a structurally higher-variance design than either A (no answer-time
LLM) or C (few LLM calls, deterministic geometry between them) on the points that matter most.

**B3 — Marker extents from a lidar-frustum fit will systematically lose IoU on the wall-mounted
and thin objects that populate the question set — B admits this is its "weakest scored output"
(§6).** Object-reference is scored by GT-box overlap (upstream_notes §3), and the training noun
inventory is thick with wall/thin items: wall lamp, picture, photo, framed records, calligraphy
painting, mirror, decal (question_analysis §4). B's MARK → single-frustum cluster fit (§1.3, §3.2)
gets sparse lidar returns on exactly these, while A and C both accumulate *tracked* multi-frame
instance point clouds and fit a trimmed AABB (proposal_A §3.2; proposal_C §4.2), which is
strictly tighter on sparse-return objects. B trades 23.5% of the points' precision for architectural
simplicity it doesn't need — the tracked-instance map is cheap and B already computes per-object
map coordinates for its ledger (§3.3), so it is *most of the way* to the better box and stops short.

**B4 — Over-engineering relative to points: the annotated-panorama discrete-action agent loop is
heavy machinery whose marginal value over "plan + deterministic execute" is unproven on
allocentric-only questions.** B renders numbered discs onto a 1536×512 pano every step, runs
RECALL/LOOK/MOVE/MARK/COUNT_TALLY as a full agent (§1.2, §5.2). But question_analysis §5 shows
*zero* egocentric phrasing — every relation is object-to-object, resolvable from a map without a
look-around agent deciding "where to gaze." The agent-loop's differentiator (interactive visual
gazing) is exactly the capability the question distribution does not require. For 70.6% of points
that decompose into ordered anchor-grounding (question_analysis §7), a plan-once-then-ground
approach (which A, B-3.1, and C all describe) does the work; the surrounding agent apparatus is
latency-and-variance cost with thin marginal points.

---

## 3. What I would steal

- **From B: dual-provider API failover (proposal_B §4.1).** C's fallback is a single local Qwen-7B;
  adding a *second* frontier API before dropping to local is strictly better for the common case
  (one provider throttles) at ~zero effort — a config string and a try/except. Integrate as: API-1
  → API-2 → local-7B, in C's reasoner (proposal_C §5), keeping the local model as the true offline
  floor B lacks. Pure upside, costs half a day.
- **From B: drawing candidate waypoints as numbered markers ON the panorama for the selection call
  (§5.2).** Parse-proof discrete action space beats coordinate lists for the few LLM calls C does
  make. C can adopt this narrowly — for frontier selection only, not as a full agent loop — getting
  the grounding benefit without B's per-step latency tax.
- **From B: the object-dimension sanity table for Marker extents (§3.2).** A cheap plausibility
  gate on C's fitted AABB ("pillow ≈ 0.5×0.4×0.15 m; reject a 2 m fit") that catches
  see-through-window ghost clusters. Two hours of work, direct IoU protection on 23.5% of points.
- **From A: fitting the Marker box to trimmed instance points at the 2nd/98th percentile per axis
  (§3.2).** C says "padded ~10%" (proposal_C §4.2); A's percentile-trim is the more principled
  tightness play against fusion outliers and is measurably better for an IoU metric. Adopt A's
  trim, keep a small pad only as a floor.
- **From A: the explicit NOT_FOUND → fallback ladder (relax attributes → relax weakest relation →
  category-only → count-unfiltered) (§6 #6).** C has a watchdog floor but A's *graded* relaxation
  ladder is a more points-preserving degrade than C's "best-guess instance." Fold A's ladder into
  C's watchdog.

---

## 4. Non-negotiables (must survive any merge)

- **The watchdog answer-floor / "the floor is not zero" scaffolding (proposal_C §4.5, row 1 of the
  points table §1.2).** Under relaunch-per-question, no-retry, partial-credit scoring, the single
  highest-return line of code is the one guaranteeing *every* question emits a plausible answer
  before T-30s regardless of internal state. It is the difference between a graceful 40/51 and a
  catastrophic 12/51 when any subsystem hangs. Both A (§4 hard gate) and B (§4 watchdog) independently
  converged on this — it is not C-specific in spirit, but C's explicit points-floor framing (guessed
  marker + modal integer + drive-at-best-anchor as a *designed* floor, not an afterthought) must be
  the merged system's backbone. Everything else is upside on top of a guaranteed floor.

- **The expected-points model as the prioritisation instrument (proposal_C §1.2).** The pts/day
  table is *why* we cut fine-tuning, cap numerical effort, and gate an MVS at Aug 3. Any merge that
  drops the explicit points-per-effort ledger will drift into building the interesting subsystem
  instead of the valuable one — A over-invests in avoid-corridor geometry (3 questions) and B in the
  agent loop (unneeded for allocentric questions) precisely because neither priced the subsystem
  against its marginal points first. The ledger is the discipline that prevents both.

- **The interleaved explore-execute policy for instruction-following (proposal_C §3, §4.1).** With a
  360° sensor suite, driving toward sub-goal 1 *is* exploration; waiting for a complete map before
  moving wastes the clock and the early-finish bonus on 70.6% of the points. A executes only after
  terminal-goal refs resolve (§4); B pipelines travel and inference but still hunts each anchor
  sequentially (§3.1). C's interleave — begin executing while continuing to scan for later anchors
  en route — is the tightest use of the budget on the highest-value type, and must survive.

---

## 5. Concessions (where a rival exposed a real weakness in C)

- **Training-distribution overfit — conceded, and both rivals share it, but A's framing is
  sharper.** C's §6.2 already names the correlated-risk problem, but A's weakness section makes the
  more honest structural point I under-stated: *fixed-threshold predicates have no reasoning
  fallback for borderline held-out cases.* C's toolbox is also threshold-based (near/on/between,
  proposal_C §4.2) — I claimed the LLM-tool-calling layer hedges this, but the LLM only *chooses*
  which threshold predicate to call; the threshold itself is as brittle as A's. I overclaimed C's
  robustness here. Real fix to fold in: scale-adaptive thresholds (A's §3.0b `near` = max(1.2 m,
  0.6× footprint diagonal)) which C did not specify — steal it.

- **C's early-answer margin rule is under-specified next to A's concrete gate.** C says "distance-
  ratio ≥1.5" and "45 s no change" (proposal_C §4.0); A gives a fully operational gate — "winner
  beats runner-up by ≥25% AND every referenced instance has ≥3 observations" (proposal_A §4). The
  observation-count condition is a real robustness lever C omitted: answering on a 1-observation
  instance is how you bank a bonus on a mis-detection. C should adopt the min-observation gate.

- **C leans harder on the API than I admitted.** B's existential-dependence critique lands partially
  on C too: C's reasoner does 3–6 API calls/question and only fails over to a weaker local 7B whose
  IF-parsing quality is "untested until week 3" (proposal_C §6.2). B is more API-exposed, but C is
  not API-*free* the way A's answer path is. A's single-parse design genuinely needs the network
  less. Conceded: C's schedule must move the local-fallback quality test earlier than week 3.

---

**Single strongest attack — A:** the single-parse DSL removes answer-time reasoning on the exact
multi-constraint, implicature-laden questions where compositional slip is the organizers' own
documented failure mode, systematically bleeding the 70.6% of points that are instruction-following
(A1).

**Single strongest attack — B:** runtime API dependence is an existential, un-hedgeable failure —
a dark eval network darkens both providers and collapses B to near-zero on all six 6-point questions
at once, a risk B itself concedes is "not hedgeable from inside the bet" (B1).

**#1 non-negotiable:** the watchdog answer-floor / "the floor is not zero" scaffolding — under
no-retry, partial-credit, relaunch-per-question scoring, guaranteeing every question emits a
plausible answer before the clock expires is the highest points-per-line insurance in the entire
design, and it is the backbone the rest of the score sits on.
