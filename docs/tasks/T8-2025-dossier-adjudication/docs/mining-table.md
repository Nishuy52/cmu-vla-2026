# Mining table: 2025 dossier priors vs implementation

Mechanical extraction (delegated scout, 2026-07-14) of concrete
thresholds and named failure modes from docs/prior_art/ dossiers,
compared against src/core/geometry/toolbox.py constants and
docs/architecture.md decisions. Facts only; verdicts live in
../verdicts.md (adjudication pass).

## Table A: Threshold priors

| Parameter | Dossier value + source | Our value + source | Status |
|---|---|---|---|
| near (proximity) | VLA-3D: `euclidean(center_a,center_b) < near_thresh * region_volume`, `near_thresh = 0.01` - prior_art/vla_3d.md:196-198 | `max(near_floor=1.2, near_scale=0.6 * footprint_diag(b))`, AABB-gap based - toolbox.py:40-41, :137-140, :201-209 | CONFLICT (functional form) |
| near (SORT3D dead variant) | `find_near_old` (unregistered): `near_thres = 1` m, bbox-corner proximity - prior_art/sort3d.md:212-223 | n/a | absent both (dead code upstream) |
| near (CopyPasta T1) | `clamp(median_bbox_diag, 0.4, 1.2) * 0.8` - prior_art/2025_3rd_copypasta.md:169-170 | `max(1.2, 0.6*diag)` - toolbox.py:40-41 | agree in spirit (scale-adaptive) |
| near (KAIST-ISE) | fixed `near_dist = 0.8` (online) / `0.75` (rich) - prior_art/2025_kaist_ise.md:261, :266 | scale-adaptive, no fixed radius | conflict (form), team prior only |
| on: vertical gap | VLA-3D `on_thres = 0.01` m directional z-band - prior_art/vla_3d.md:161, :174-176; SORT3D same - prior_art/sort3d.md:283-284 | `on_vert_tol = 0.15` m symmetric, "(invented)" - toolbox.py:43, :159-178 | CONFLICT (15x looser, symmetric) |
| on: footprint overlap | VLA-3D `vertical_iom > 0.5` - prior_art/vla_3d.md:160-161; SORT3D 0.5 - sort3d.md:248; KAIST-ISE 0.5 - 2025_kaist_ise.md:223-226 | `on_min_overlap_frac = 0.30` "(invented)" - toolbox.py:44 | CONFLICT (3 sources converge on 0.5) |
| above/below: overlap gate | VLA-3D `vertical_iom > 0.5` - vla_3d.md:156-160; KAIST-ISE `overlap_ratio > 0.2` - 2025_kaist_ise.md:217-218 | boolean `footprints_overlap`, no fraction gate - toolbox.py:244-269 | CONFLICT (no gate at all) |
| above/below: z-gap | KAIST-ISE `above_gap = 0.20/0.35` as MINIMUM gap - 2025_kaist_ise.md:217-218, :261, :266 | `above_gap_max = 3.0` as CAP "(invented)" - toolbox.py:47 | conflict (inverse roles; team prior) |
| between | VLA-3D `overlap_thres=0.3, symmetry_thres=0.5, distance_thres=1, between_iom=0.5` - vla_3d.md:240-248; SORT3D same names but `between_iom=0.05` - sort3d.md:202-206 | capsule test, radius = max anchor half-width, no IoM gate - toolbox.py:222-241 | CONFLICT (form; sources also disagree with each other) |
| closest/farthest ambiguity | VLA-3D `ordered_thres = 0.2`: statement suppressed if consecutive-rank distance gap * avg_length <= 0.2 - vla_3d.md:220-229 | `superlative_margin_frac = 0.25` gates early-answer only, no suppression - toolbox.py:50; architecture.md:32 | analogous, different mechanism |
| avoid inflation | not found in any dossier | `avoid_inflate = 0.25` m - toolbox.py:49 | absent in dossiers |
| in/containment | VLA-3D `relate_in` = strict signed-distance face-plane containment - vla_3d.md:185-188 | `in_containment_frac = 0.60`, `in_vert_slack = 0.10` "(invented)" - toolbox.py:45-46 | different form, no numeric prior |
| color vocabulary | closed 15-color set - vla_3d.md:105-107, :441-442 | CSV `object_color_schemeN` consumed verbatim - loader.py:155-162 | agree (inherited) |
| color dominance floor | color counts only if >= 0.1 * total points - vla_3d.md:110-114 | trusted from CSV, not recomputed | agree (inherited) |
| big/small basis | relative `largest_face_area`, > 1.2x next same-class - vla_3d.md:305-309; SORT3D: size = largest-face area, not volume - sort3d.md:181-184 | absolute AABB volume cutoffs 0.5 / 0.02 m^3, class-agnostic - loader.py:51-52, :165-172 | CONFLICT (2 sources converge) |
| left/right/front/behind | SORT3D: stubbed `pass` - sort3d.md:270-279; KAIST-ISE: implemented, `lr/fb_thresh = 0.20/0.05` - 2025_kaist_ise.md:227-234 | no such predicate (Pred enum) - plan_schema.py:24-34 | absent in ours AND in VLA-3D question vocabulary |

## Table B: Named failure modes

| Failure mode | Source | Addressed in our architecture? |
|---|---|---|
| Open-vocab detection quality is the ceiling | prior_art/README.md:531-536 | Partially - checkpoint 2 detector-miss recovery (architecture.md:30); residual risk #2 (:148-149) |
| Relation-threshold ambiguity (between/closest/near) | prior_art/README.md:537-543 | Structurally - deterministic toolbox over LLM freestyling (architecture.md:55-58) |
| Exploration cost under 10-min clock | prior_art/README.md:544-551 | Addressed - frontier planner + per-type budget (architecture.md:106-113) |
| Color/attribute extraction | prior_art/README.md:552-555 | Partially - captioning is stretch/cuttable (architecture.md:122) |
| IF hardest + highest value | prior_art/README.md:556-561 | Addressed - deepest machinery (architecture.md:95-98) |
| Numerical exactness binary; dedup matters | prior_art/README.md:562-565 | Addressed - NMS-deduplicated set cardinality (architecture.md:102-104) |
| LLM hallucination / format drift | prior_art/README.md:566-568 | Addressed - schema-validated plan + fallback ladder (architecture.md:41) |
| SORT3D: wrong-anchor filtering + pragmatics miss ("rightmost pillow" implies "on the bed") | sort3d.md:489-495 | Partially - verification checkpoint; implicature named residual risk #3 (architecture.md:150-151) |
| SORT3D: LLM variance up to 6% between trials | sort3d.md:464-468 | Partially - self-consistency x3 optional (architecture.md:88, :139) |
| SORT3D: front/behind advertised but stubbed | sort3d.md:270-279 | Moot - VLA-3D questions lack the relation family |
| CopyPasta: IF only 35.2% sub-goals reached (worst type) | 2025_3rd_copypasta.md:242-248 | Addressed as priority signal |
| CopyPasta: reprompt-until-100 as only IF recovery | 2025_3rd_copypasta.md:333-336 | Addressed differently - bounded event-triggered checkpoints (architecture.md:90) |
| ReasonX: whole grounding depends on GT `/object_markers` | 2025_2nd_reasonx.md:284-298 | Addressed by design - no GT-topic dependency (architecture.md:46-52) |
| ReasonX: zero time-budget instrumentation | 2025_2nd_reasonx.md:313-318 | Addressed - T-90s/T-30s watchdog (architecture.md:106-113) |
| URL-KAIST: GT markers fused with priority over vision | 2025_4th_url_kaist.md:394-402 | Addressed by design (as above) |
| URL-KAIST: raw LLM Python exec()'d, no sandbox | 2025_4th_url_kaist.md:225-232 | Addressed - typed JSON, never executable code |
| URL-KAIST: frontier explorer ignores avoid-filtered map | 2025_4th_url_kaist.md:200-205 | PARTIAL GAP - avoid capsules hard for path exec; frontier-scoring avoid-awareness not explicit (architecture.md:46-52) - needs a direct check |
| URL-KAIST: per-scene-tuned detector classes, generalization risk | 2025_4th_url_kaist.md:113-127 | Partially - residual risk #1 accepted knowingly (architecture.md:145-147) |
| KAIST-ISE: two undocumented divergent threshold profiles | 2025_kaist_ise.md:256-272 | Addressed - single DEFAULT_THRESHOLDS source of truth (toolbox.py:32-53) |
| KAIST-ISE: robot-relative relations silently broken | 2025_kaist_ise.md:236-246 | Moot - no such predicates |
| KAIST-ISE: track geometry frozen at first observation | 2025_kaist_ise.md:279-283 | Addressed - percentile-trimmed AABB over tracked clouds (architecture.md, marker-extents decision) |

## Table C: Conflicts requiring adjudication

| # | Conflict | Dossier citation | Ours |
|---|---|---|---|
| 1 | `near` functional form: region-volume-scaled centroid distance vs footprint-diag-scaled AABB gap | vla_3d.md:196-198 | toolbox.py:40-41, :137-140, :201-209 |
| 2 | `on` footprint-overlap gate: 3 sources converge on 0.5; ours 0.30 | vla_3d.md:160-161; sort3d.md:248; 2025_kaist_ise.md:223-226 | toolbox.py:44 |
| 3 | `on` vertical tolerance: directional 1 cm band vs symmetric +/-0.15 m | vla_3d.md:161, :174-176; sort3d.md:283-284 | toolbox.py:43 |
| 4 | `above`/`below` overlap-fraction gate: required (0.2-0.5) vs absent | vla_3d.md:156-160; 2025_kaist_ise.md:217-218 | toolbox.py:244-269 |
| 5 | `between` form: projected 1D frame + 4 gates vs single capsule test (sources internally disagree: between_iom 0.5 vs 0.05) | vla_3d.md:230-261; sort3d.md:194-211 | toolbox.py:222-241 |
| 6 | big/small: relative per-class largest-face-area (>1.2x) vs absolute class-agnostic volume | vla_3d.md:304-309; sort3d.md:181-184 | loader.py:51-52, :165-172 |
| 7 | `above_gap`: dossier minimum-gap gate vs our (unenforced) maximum cap | 2025_kaist_ise.md:217-218 | toolbox.py:47 |
