# Red-team review — object-reference head & LLM checkpoint protocol

**Date:** 12 Jul 2026
**Scope:** object-reference resolve/marker path (`src/core/geometry/`, `src/core/heads/object_ref.py`, `src/core/perception/scene_index.py`) and the live checkpoint protocol CP2–CP5 (`docs/checkpoint_design.md`, `src/core/checkpoints/`), against the GT battery (`reports/gt_battery_full_2026-07-11/gt_battery_report.md`) and the question taxonomy (`docs/question_analysis.md`).

**Verdict.** The object-reference head has a structural hole that the GT battery already exposed but the topline number hides: nested disambiguator constraints — the *dominant* multi-constraint OR form ("the bowl **on the table closest to** the folding screen") — are silently discarded by the resolve ladder, and both wrong-instance results in the battery are exactly this form (2/2 of the scored multi-constraint questions, IoU 0.000). Reproduced deterministically on a 5-object synthetic scene: the head returns the bowl on the *far* table with an empty audit trail. Around that hole sit three amplifiers: a typo tier that pollutes the candidate pool even when exact label matches exist, attribute mentions that silently collapse superlatives to lowest-instance-id ordering, and a live label-matching synonym table (4 groups) that is far thinner than the vocabulary drift the battery itself measured (24/30 non-matches; the vocab bridge that fixed 6→7 lives only in the *scorer*, not the live pipeline). On the checkpoint side, CP4 — the checkpoint that is supposed to catch precisely this clause-slip — cannot: its re-resolve hook rebuilds the identical TargetSpec so it is a token-costing no-op, and the legacy bool seam actively *inverts* the design's "fallback = keep winner" rule, so a hallucinated "neither" verdict swaps to an unverified runner-up the model never saw. The JSON-parse/fallback plumbing itself is solid (balanced-brace extraction, one repair, total validators, deterministic fallbacks) — with one systemic exception: repair calls and the parse-ladder tiers run outside the timeout envelope, so a hanging (rather than refusing) network erodes the 45-s floor reserve.

---

## F1 — Nested disambiguator constraints are silently dropped by `resolve()` (wrong instance on the dominant OR form)

- **Severity: ~20–34 pts.** 17/30 OR questions are multi-constraint (`docs/question_analysis.md` §5: 57%), and the canonical hard form chains a support relation with a nested proximity superlative. Each is 2 pts and IoU-scored against the *one* correct instance, so picking the wrong sibling scores ~0. Both battery wrong-instance results are this form. Also leaks into numerical (4 nested counting questions, §6) and IF anchor grounding (same `_resolve_anchor`), so the true exposure is larger than OR alone.
- **Evidence:**
  - `src/core/geometry/toolbox.py:358-367` — `_resolve_anchor` docstring: "**ignores disambiguator nesting** … nested disambiguation is a resolver concern handled by resolve() on the top-level target". But `resolve()` never recurses: `_superlative_clause` (`toolbox.py:430-434`) scans only `target.clauses` top level, and `_eval_clause` (`toolbox.py:382`) resolves anchors existentially over *all* noun matches.
  - The parser correctly produces the nesting: `regex_tier.py:127-135` builds `Anchor(disambiguator=closest_to(...))`; the API prompt teaches the same shape (`prompts.py:248-249`, rule 4). So parse is fine — the toolbox drops it.
  - Reproduced (synthetic scene, this review): question "Find the bowl on the table closest to the folding screen" parses to `bowl / on(table, disamb=closest_to)` and `resolve()` ranks the bowl on the **far** table first (`winner id 1, correct 2`), audit trail **empty** — no relaxation was even recorded, because nothing failed; the constraint simply never applied.
  - Battery: `gt_battery_report.md` rows chinese_room ("bowl on the table closest to the folding sc…", **IoU=0.000**) and home_building_2 ("speaker on the TV cabinet closest to the po…", **IoU=0.000**) — the only two scored multi-constraint questions, both wrong instance. The 4 perfect IoUs are all single-relation or top-level-superlative forms ("clock on the TV cabinet", "picture closest to the bench", "vase closest to the guitar", "potted plant between a vase and the cabinet").
- **Attack scenario:** any scene with ≥2 support anchors of the same class ("two tables, a bowl on each"). The target passes `on(table)` existentially for both bowls; ranking falls to `_stable_by_id` (`toolbox.py:538-539`) — i.e. whichever bowl was *detected first* wins. The eval scenes are furnished rooms; duplicate tables/cabinets/sofas are the norm, not the edge case.
- **Proposed fix (design):** make `_resolve_anchor` honour `anchor.disambiguator`: (a) non-superlative disambiguator → filter anchor candidates by evaluating the clause with the anchor as subject; (b) superlative disambiguator → rank anchor candidates and keep the argmin/argmax **only**; then evaluate the outer clause against the narrowed anchor set. Additionally, when a nested superlative exists and the outer clause still leaves >1 survivor, break the tie by the nested metric rather than instance_id. Record a `Relaxation` when a disambiguator has to be dropped so the audit trail is honest.
- **Cost: M** (one function + recursion guard + goldens for the two battery failures).

## F2 — Live label matching is far thinner than the measured vocabulary drift; the vocab bridge only fixes the *scorer*

- **Severity: ~10–30 pts (upper bound is soft).** The battery's 24/30 "no GT target matched" rows are partly a dev-harness artifact (question nouns vs *annotation* vocabulary), but the same drift class exists on the live path: question nouns vs *detector* labels. There, matching is `BasicSceneIndex.by_label` with exactly **4 synonym groups** (`scene_index.py:29-34`: refrigerator/fridge, sofa/couch, television/tv, picture/photo). The drift the battery actually observed — bedside table↔night stand, potted plant↔plant, beer bottle↔bottle, paper cup↔cup, computer monitor↔monitor, wall lamp↔lamp — was curated into `core/groundtruth/vocab_bridge.py:67-84`, which is imported only by `core/groundtruth/scoring.py` (offline scorer). Nothing in `perception/` or `geometry/` consults it. So "scoreable 6/30" is *both* an artifact *and* a live warning: the harness proved this drift is real and dense, and the live matcher has no answer to it.
- **Evidence:** `scene_index.py:29-34` vs `vocab_bridge.py:67-84`; grep shows `vocab_bridge` imported only from `core/groundtruth/`. `docs/question_analysis.md` §4: 114 noun tokens incl. fine-grained classes ("hookah", "map wall decal", "framed records") that an open-vocab detector will emit under different surface forms.
- **Attack scenario:** eval question "Find the bedside table farthest from the window" (verbatim training form, hotel_room_1); detector emits `night stand`. `by_label("bedside table")` → exact none, synonym none, typo none (edit distance 8) → empty pool → fallback ladder → `category_only` over an **empty** base → head returns None/floor marker. 0/2 pts on a question the map actually contained.
- **Proposed fix (design):** move the bridge into the live path — fold `vocab_bridge._bridge_map()` into `BasicSceneIndex.by_label`'s synonym tier (it is already normalised and bidirectional), and generalise: head-noun match ("X table" matches "table"-headed labels, "beer bottle"→"bottle") as an explicit *lower-priority* tier below exact/synonym, above typo. The bridge module's honest-none discipline (never hyponyms) transfers as-is.
- **Cost: S** (import + one tier in `by_label`; the table already exists and is scene-verified).

## F3 — Typo tier pollutes the candidate pool even when exact matches exist

- **Severity: ~6–12 pts (OR wrong-instance + numerical over-count).** `by_label` returns `exact + syn + typo` **concatenated** (`scene_index.py:154-155`), and `resolve()` treats the pool uniformly (`toolbox.py:353-355`) — ranking is by superlative distance or instance_id, never by match tier. Levenshtein ≤2 on a 100+-noun vocabulary is riddled with collisions.
- **Evidence (verified this review):** `_levenshtein("door","floor")=2`, `("book","bowl")=2`, `("stool","stove")=2`. Live repro: `by_label('door')` → `[door, floor, book]`; `by_label('book')` → `[book, door, bowl]`. A *floor* instance (huge AABB, near everything) entering a "door" pool is poison for `near`/`closest_to` clauses; counting (`toolbox.py:545-559`) counts resolve survivors, so typo cousins inflate numerical answers — consistent with the battery's numerical over-count signal (independent agreement 15–27%, `gt_battery_report.md` topline).
- **Attack scenario:** "Find the book closest to the sofa" in a scene with books and a bowl; the bowl sits nearer the sofa; the bowl wins the superlative and is published. Wrong instance, 0/2.
- **Proposed fix (design):** (a) tier short-circuit — return typo matches only when exact+synonym tiers are empty; (b) length-scaled tolerance — distance ≤2 only for nouns of length ≥6, ≤1 for length 4–5, exact only for ≤3; (c) surface the match tier on the record so `resolve` can prefer exact-tier candidates before superlative ranking.
- **Cost: S.**

## F4 — CP4 legacy bool seam inverts the fallback rule: a wrong "neither" makes the answer worse than not verifying

- **Severity: up to ~8–12 pts (it guards all 60 OR pts; it can flip the 4 currently-perfect wins).** The design (`docs/checkpoint_design.md` §CP4) and the rich path (`verification.py:188-204`) treat `neither`-without-actionable-info as "keep winner". The deployed-compatible seam does the opposite.
- **Evidence:**
  - `verification.py:288` — `return obj["verdict"] == "confirm"`: **both** `runner_up` and `neither` map to False.
  - `object_ref.py:213-224` — on False the head walks to candidate 2 and asks again; but the ledger caps `verification` at 1 (`budget.py:97-104`), so the second `guarded_call` is denied → returns None → seam returns True (`verification.py:283-284`) → **candidate 2 is auto-kept unverified**. Net deployed semantics: any non-"confirm" verdict silently swaps to the runner-up.
  - `verification.py:279` — the seam prompt hardcodes `Runner-up: (none)`; the model is asked to choose `runner_up` for a candidate it is explicitly told does not exist, and the candidate it actually swaps to was never shown to it.
  - The prompt (`verification.py:55-64`) demands "Check EVERY requirement … including any the clause list may have MISSED" while providing **no scene facts beyond the winner's clause table** — the model cannot verify a suspected miss, so the calibrated response to any doubt is `neither`. Doubt is common on relaxed-ladder answers (audit shows dropped clauses), which are exactly the answers where the deterministic winner is still the best available.
- **Attack scenario (concrete malformed-ish but schema-valid reply):** winner is correct; model returns `{"verdict":"neither","missed_constraint":"the table must be the one closest to the folding screen","reason":"cannot confirm the table constraint from the facts"}`. Seam → False → swap to candidate 2 (the *wrong* bowl) → unverified keep. A perfect 2-pt answer becomes 0. This is strictly worse than disabling CP4.
- **Proposed fix (design):** wire the **rich** verifier through `HeadState.verifier` (the guard `runner_up is None → keep` already exists at `verification.py:191-194`); if the bool seam must stay, map `neither → True` (keep, per design fallback) and pass the real runner-up summary instead of "(none)". Improved prompt wording (replace the last instruction block):

  > "Check every requirement in the question against the computed facts. Reply `confirm` if the Selected instance satisfies them all. Reply `runner_up` ONLY if the Runner-up shown above clearly satisfies a requirement the Selected instance fails — never if Runner-up is (none). Reply `neither` ONLY if a specific requirement is contradicted by the computed facts for BOTH candidates; in that case set `missed_constraint` to the single relation word plus anchor noun (e.g. `closest_to folding screen`), copied from the question. If the facts are merely insufficient to decide, reply `confirm`. JSON only: {…}"
- **Cost: S** (wiring + two-line seam change + prompt constant).

## F5 — CP4's missed-constraint re-resolve is a no-op by construction, so the clause-slip guard cannot fire

- **Severity: indirect — it is the designated counter to F1 (the documented failure mode), and it does not work; also wastes 1 ledger call + ~20 s worst-case.**
- **Evidence:**
  - `object_ref.py:190-196` — `resolve_again` rebuilds `TargetSpec` with `clauses=list(base_target.clauses)` **unchanged**: the missed constraint is appended to `plan.notes` (audit only), never synthesized into a clause. The re-run `resolve` returns the identical ranking.
  - `object_ref.py:199-205` + `_contradicts_constraint` (`object_ref.py:250-260`) — the skip test requires the *entire* missed string (`token = missed.strip().lower()`) to appear as a substring of a **FAIL**ing explanation. Survivors by definition passed all applied clauses (`toolbox.py:524-535`), and after `category_only` the matrix is empty (`toolbox.py:489-490`, `hard_clauses = []`) — so FAIL rows on survivors essentially never exist, the loop matches nothing, and `res.candidates_ranked[0]` (the same winner) is returned. The rich-path outcome is `("re_resolve", same_winner)` — a cosmetic action.
  - Nothing tells the model what shape `missed_constraint` should take (prompt at `verification.py:55-64`, schema at `schemas.py:106-116` allows any string), so even a fixed matcher would receive free-form sentences.
- **Attack scenario:** F1's scene; CP4 correctly diagnoses "neither, missed: closest to the folding screen". Rich path: re-resolve returns the same wrong bowl; answer unchanged; the one verification call and the 90-s reserve check bought nothing. (Via the legacy seam it is worse — see F4.)
- **Proposed fix (design):** constrain `missed_constraint` to `"<pred_word> <anchor noun>"` (prompt wording in F4), then synthesize a real clause: map pred_word through the regex tier's `_REL_TOKENS`, build `Clause(pred, [Anchor(noun)])`, append to the spec copy before re-resolving. If the pred maps to a superlative, attach it as the ranking clause. Fall back to keep-winner when the string does not parse.
- **Cost: M.**

## F6 — Marker extents are validated only circularly; the "dimension sanity table" does not exist; trimmed AABBs will under-fill the GT hull on real perception

- **Severity: ~4–12 pts (2-pt cliffs at IoU 0.25 / 0.5 across all 30 OR questions once real perception is in the loop).**
- **Evidence:**
  - The four IoU=1.000 rows in `gt_battery_report.md` are GT-geometry self-matches (the report's own circularity note; instances were built from GT boxes), so marker tightness has **never** been measured against independently-produced boxes.
  - `interfaces.py:122-135` — marker = trimmed 2nd/98th-pct AABB (`scene_index.py:107-111`), centered at box midpoint. No per-class dimension clamp exists anywhere: grep for `sanity` across `src/core` finds only tracker/scene-index trim comments — the "dimension sanity table" referenced in planning does not exist in code.
  - `gt_battery_report.md` OBB→AABB note: GT boxes are axis-aligned hulls of oriented boxes — a strict *over*-approximation. Our trimmed box built from single-viewpoint lidar returns will be a strict *under*-approximation (only observed faces carry points; 2/98 trim shaves a further 4% per axis on sparse clouds). Under-box inside over-box compounds: a sofa seen from the front can easily land under IoU 0.5 (lose 1 pt) or 0.25 (lose 2).
  - The marker center is also the nav goal (`interfaces.py:91`) — a front-shell box drags the goal toward the observed face.
- **Attack scenario:** "Find the picture closest to the bench" (battery-perfect today). Real perception sees the picture face-on: depth extent ≈ a few cm of lidar noise, trimmed further; GT hull has the frame's true depth plus rotation slack. IoU falls below 0.5 despite the correct instance.
- **Proposed fix (design):** add the missing per-class dimension prior table (min/typical extents for the ~80 vocab nouns; VLA-3D stats can seed it); clamp marker extents to at least the class-min per axis, and inflate the *unobserved* axis (viewing-ray direction at last observation) toward the class-typical value. Re-run the battery with perception-produced boxes (not GT geometry) to get a non-circular IoU baseline.
- **Cost: M** (table + one clamp in `to_marker`; the battery rerun is the existing harness).

## F7 — Repair calls and parse-ladder tiers run outside the timeout envelope; a hanging network erodes the 45-s floor reserve

- **Severity: systemic, all 255 pts in the tail-risk sense; realistic loss is the floor-answer delta on any question where a call hangs.**
- **Evidence:**
  - `_runtime.guarded_call` wraps only the primary call in `call_with_timeout` (`_runtime.py:61-68`). The repair callable is invoked *afterwards*, inside `schemas.parse_with_repair` (`schemas.py:187`), directly — no timeout, no ledger gate, no record. Every checkpoint forwards `repair` into that path (`verification.py:180`, `miss_recovery.py:114`, etc.).
  - The parse ladder's `_attempt` calls `fn(messages)` raw (`ladder.py:88-98`); its 45-s cap is only checked **between** attempts (`ladder.py:55-77`), so one blackholed TCP connection blocks the whole tier indefinitely unless the injected ChatFn embeds its own timeout (not verifiable from `core/` — see below).
  - Ledger math: `CallLedger.allow` requires `remaining ≥ 45 s` (`budget.py:23,137-148`). A CP4 admitted at T-46 s can spend 20 s (primary, bounded) + unbounded repair — the "graceful degradation" claim (api→api2→local→regex parse ladder analogy) holds for *refusing* networks but not *hanging* ones.
- **Attack scenario:** eval-day network degrades to silent packet drop (common failure of NATed venue networks, unlike the clean `ConnectionError` the fallbacks were built for). CP4 fires at T-60 s; primary times out at 20 s; repair hangs 60+ s; the head's `verify()` blocks past the T-30 watchdog. Whether the floor still publishes depends on FSM threading (unverified) — if `verify` runs on the control thread, the question times out with no answer.
- **Proposed fix (design):** run the repair round under the same `guarded_call` envelope (name it `<checkpoint>.repair`, tier `repair`, recorded); wrap ladder `_attempt` in `call_with_timeout(min(remaining_tier_budget, 20 s))`; make `Ledger.allow` subtract a worst-case call cost (timeout + repair timeout) instead of a flat 45-s floor.
- **Cost: S.**

## F8 — CP2's provisional instance can become the published OR marker with unvalidated geometry

- **Severity: ~2–6 pts, plus wasted drive time.** Bounded by the cap (1 call/question) and by only firing when the noun has 0 instances — but that is exactly the state where the floor would otherwise publish *nothing*, so a hallucinated bad box replaces an honest miss with a confident wrong answer, and in IF it redirects a 6-pt leg.
- **Evidence:**
  - `schemas.validate_miss_recovery` (`schemas.py:119-133`) accepts any 4 numbers as `bbox_hint` — no x1<x2/y1<y2, no non-negativity, no tile-bounds check (tiles are ≤640 px; the prompt never tells the model the tile dimensions, `miss_recovery.py:47-51`).
  - No confidence floor: `present=true, confidence=0.05` still yields `action="provisional"` (`miss_recovery.py:118-139`).
  - The ≥3-obs cap guards only the *early-answer* gate (design risk note; `miss_recovery.py:18-20`). `resolve()` never consults `n_obs` or `score` (`toolbox.py:437-521`), and `ObjectRefHead.advance` publishes `candidates_ranked[0]` unconditionally (`object_ref.py:92-101`) — so a fused n_obs=1 hallucination is immediately the best marker for a noun with no other instances, and `verify()` will commit it.
- **Attack scenario:** "Find the hookah …" in a hookah-less test scene. At 60% budget CP2 fires; the VLM pattern-matches a vase, returns `{"present":true,"tile":2,"bbox_hint":[630,10,700,90],"confidence":0.3}` — x2 beyond the 640-px tile. The cast ray lands on a wall; a provisional instance is fused there; the marker publishes a box on empty wall. Score 0/2 *and* the robot detoured to it.
- **Proposed fix (design):** validate bbox ordering/bounds in `validate_miss_recovery` (pass tile w/h in, reject out-of-bounds); require `confidence ≥ 0.5` for provisional; tag provisional instances and have `ObjectRefHead.verify` refuse to commit a marker whose winner is provisional-only unless it was re-observed (n_obs ≥ 2) — the re-navigation step CP2 already schedules is the natural chance to upgrade it. Prompt addition: "Tiles are {w}x{h} pixels; coordinates must lie within the named tile. If unsure, answer present=false."
- **Cost: S.**

## F9 — CP3 demotes on a single 0.6-confidence mismatch, contradicting its own "map beats one report" rule; `actual_label` is collected but unused

- **Severity: IF-side (~shared with the IF facet): a wrong demotion re-plans a 6-pt leg to the runner-up mid-drive.**
- **Evidence:** `anchor_confirm.py:33` sets `MISMATCH_MIN_CONFIDENCE=0.6` while the design text (`checkpoint_design.md` §CP3, and the module docstring itself) says one mismatch report is *weaker* evidence than the tracked map. VLM self-reported confidences are known to cluster ≥0.8, so in practice every mismatch demotes. `actual_label` (`anchor_confirm.py:96-101`) is never compared to the anchor noun — a crop reply of `match=false, actual_label="couch"` for anchor "sofa" (same object, synonym) still demotes.
- **Attack scenario:** IF leg to "the sofa near the window"; arrival crop is off-center (projection error), model reports `{"match":false,"actual_label":"couch","confidence":0.85}`; correct anchor demoted, leg re-planned to a farther sofa; trajectory score drops on a leg that was already correct.
- **Proposed fix (design):** demote only when `match=false` AND `normalize_label`/bridge says `actual_label` is a *different* class from the anchor noun AND confidence ≥0.8; treat synonym/null `actual_label` as confirm. Prompt improvement: "First name the object centred in the crop (`actual_label`), then judge `match`. If the crop is ambiguous, blurred, or the object is cut off, answer match=true with low confidence."
- **Cost: S.**

## F10 — Prompt/cost hygiene: parse system prompt already exceeds the ≤2k-token budget; default `.npy` image encoder is a live-eval landmine; CP5 has no abstain

- **Severity: low points, but each is a silent-degradation trap.**
- **Evidence:**
  - Parse (CP1) `SYSTEM_PROMPT` measures 9,226 chars ≈ 2.3k tokens (computed this review) *before* the user message — above the design's ≤2k-in rule (`checkpoint_design.md` "Shared rules"); with cap 2 parse calls + repairs it is the dominant token spend. CP2–CP5 prompts are small (≤~200 tokens each + question/fact table) and within budget.
  - `_vision.default_encode_fn` emits raw `.npy` bytes (`_vision.py:21-31`); a real vision API will reject them. Failure is graceful (fallback), but if Phase-2 wiring forgets the JPEG `encode_fn` swap, CP2/CP3/CP5 silently *never* work at eval while all tests stay green — there is no wiring-time assert.
  - CP5's prompt (`frontier_select.py:40-43`) forces a 1–N choice with no abstain; a model that answers "none of these help" in prose fails validation → geometric fallback (fine), but a model that dutifully picks a random disc *passes* validation and overrides a better geometric prior.
- **Proposed fix (design):** trim the parse schema (drop `$schema`/descriptions, `indent=None`) to get under budget; add a startup assert in the ROS adapter that the bound `encode_fn` is not `default_encode_fn` when a network provider is configured; CP5 prompt: add `"choice": 0` = "no direction clearly helps — use your own heuristic" mapped to fallback.
- **Cost: S.**

---

## Is "scoreable on 6/30" hiding eval risk?

Split verdict, per the evidence above:

- **Artifact component (real):** the 24 non-matches are question-noun vs *annotation*-vocabulary drift in the dev harness. The live eval is scored by the organizers against Unity GT boxes — annotation vocabulary is irrelevant there. The vocab bridge (6→7) correctly attacks only the harness.
- **Hidden risk component (also real):** (1) the same drift class recurs live as question-noun vs *detector*-label drift, and the live matcher has 4 synonym groups vs the ≥7 drift pairs the harness already proved exist (F2); (2) the battery validated instance *selection* on only 6 questions, and on the multi-constraint subset the measured wrong-instance rate is 2/2 (F1); (3) all IoU numbers are GT-geometry self-matches, so marker tightness is unmeasured (F6). The 6/30 headline therefore understates how little of the OR head's live behaviour has actually been exercised.

## What I could not verify

- **Production checkpoint wiring:** `heads/factory.py` defaults every seam to None and no production binding exists in the repo (Phase-2 `ros_adapter` work), so whether CP4 ships as the rich verifier or the inverting legacy seam (F4) is undetermined — the review treats the seam as the deployed-compatible path because the module docstring says it "keeps today's tests green".
- **FSM threading during a blocked `verify()`:** `core/fsm/controller.py` was not reviewed; whether the T-30 watchdog floor can publish while a head call hangs (F7's worst case) is unconfirmed.
- **Provider-side timeouts:** `core/llm/providers` and the `call_with_timeout` implementation were not read; injected ChatFns may embed their own socket timeouts, which would soften (not remove) F7.
- **CP2/CP3/CP5 caller-side handling:** `heads/explore_step.py` and `heads/instruction.py` were not read; F8/F9 caller behaviour is inferred from the checkpoint modules' documented outcome semantics and `factory.py` wiring.
- **Detector label vocabulary at eval:** F2's live magnitude depends on how the Phase-2 open-vocab detector is prompted (question-derived prompts would shrink the drift); no detector config exists in `src/core` to check.
- **Vocab-bridge 6→7 provenance:** confirmed via `LOG.md:298` and `test_cvsweep.py`; the cvsweep report artifact itself was not re-run.
- **Token estimates** use a chars/4 heuristic, not a provider tokenizer.
- Several file reads were transiently denied by the session harness mid-review; all were successfully retried, so no planned source went unread — the items above are scope choices, not access gaps.
