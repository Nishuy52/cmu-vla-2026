> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# VLA-3D dataset - the organizers' own scene-graph and
relation-vocabulary spec, verified against the actual
generation code

Compiled 2026-07-11. This deepens the short summary already in
`docs/prior_art/README.md` section 2. Everything below is verified
against the cloned generation source
(https://github.com/HaochenZ11/VLA-3D , commit
`a0c023c9662bd875f0b3aeff4ff60adc1991e449`, fetched
2026-07-11), not just the paper text, because the paper/README
do not state exact thresholds - the code does. Paper text and
README are cited separately from code where they differ or add
context. One quirk was independently reproduced (see the "near"
relation note) rather than asserted from reading alone.

## What it is

- Paper: "VLA-3D: A Dataset for 3D Semantic Scene Understanding
  and Navigation," Haochen Zhang, Nader Zantout, Pujith
  Kachana, Zongyuan Wu, Ji Zhang, Wenshan Wang. Presented at the
  1st Workshop on Semantic Reasoning and Goal Understanding in
  Robotics (SemRob), RSS 2024. arXiv v1 submitted 2024-11-05, no
  later version listed on arXiv as of this pass.
  Source: https://arxiv.org/abs/2411.03540
- Repo: https://github.com/HaochenZ11/VLA-3D . The README carries
  a 2025-05 notice that the dataset has since been "filtered and
  extended with misaligned grounding statements" in a follow-up,
  **IRef-VLA** (https://github.com/HaochenZ11/IRef-VLA , paper
  arXiv 2503.17406, https://huggingface.co/papers/2503.17406) -
  if a cleaner/newer statement set is wanted, check that repo too
  (not reviewed in this pass beyond the README pointer).
- Scale (per README): 7,635 scenes, 11,619 regions total, drawn
  from 6 source datasets, object counts per scene ranging 4-2,264:
  - Matterport3D: 90 scenes, 2,195 regions.
  - ScanNet: 1,513 scenes.
  - HM3D: 140 scenes, 1,991 regions.
  - **Unity: 15 scenes + 3 scenes omitted for the challenge,
    46 regions** (this is the sim engine the CMU-VLA-Challenge
    itself runs).
  - ARKitScenes: 4,494 scenes.
  - 3RScan: 1,381 scenes.
  All 6 are real-world scans except Unity, which is synthetic.
  Source: https://github.com/HaochenZ11/VLA-3D (README, "Raw
  Data" section).
- 477 unique object classes referenced across all generated
  statements (README).
- 23.5M heuristically generated relations, 9,696,079 total
  generated statements including relation synonyms; README states
  "9.6M+ unique statements (without relation synonyms) exist in
  the dataset." An automated fetch of the arXiv HTML also
  reported ">286K objects" total - this number was not
  independently cross-checked against the CSVs in this pass, flag
  as **unverified**.

## Object attributes (exact definitions)

Per-object CSV columns, verbatim from
`3d_data_preprocess/utils/headers.py` `OBJECT_HEADER`:
`object_id, region_id, raw_label, nyu_id, nyu40_id, nyu_label,
nyu40_label, object_bbox_c{x,y,z}, object_bbox_{x,y,z}length,
object_bbox_heading, object_front_heading`, then 3 repeated
blocks of `object_color_{r,g,b}N, object_color_schemeN,
object_color_scheme_percentageN,
object_color_scheme_average_distN` for N in 1..3.

- **Class mapping**: every object gets both an NYUv2 id/label and
  an NYU40 id/label. ScanNet and Matterport3D reuse the original
  authors' category-mapping tables; Unity, HM3D, ARKitScenes and
  3RScan each got a **hand-authored** mapping CSV (paths:
  `3d_data_preprocess/unity/labels/Categories_<scene>.csv`,
  `3d_data_preprocess/hm3d/category_mappings/
  hm3d_full_mappings.csv`, `3d_data_preprocess/arkit/
  arkit_cat_mapping.csv`,
  `3d_data_preprocess/3rscan/3rscan_full_mapping.csv`).
  For Unity specifically, an automated fetch of the arXiv HTML
  reported that "ground-truth semantic labels were cleaned then
  manually mapped to the class schemas by five data annotators. A
  validation round was done to standardize the labels" - this
  phrasing was not re-confirmed verbatim in this pass against the
  raw HTML, treat as **probably accurate, not independently
  re-quoted**.
- **Oriented bounding box**: center + per-axis length + a single
  `heading` angle, rotation about the Z axis only (see `rotz()` /
  `get_bbox_coords_heading()` in
  `scene_graph/bbox_utils.py`). Boxes are therefore
  Z-up-yaw-only oriented, never tilted on X/Y.
- **Dominant colors** - the exact pipeline, from
  `3d_data_preprocess/utils/dominant_colors_new_lab.py`:
  1. Convert every object point's RGB to CIE-LAB via a fixed
     3x3 matrix (not `skimage`/`colormath` - a hand-rolled
     conversion in `rgb2lab()`), returning `[b, a, L]` in that
     axis order (note: not the usual `[L, a, b]` order - a
     literal quirk in the code, matters if you index into it).
  2. Build a KD-tree over the CSS3 named-color palette (via
     `webcolors`), each anchor color's RGB also converted to LAB.
  3. Nearest-neighbor-match every object point to its closest
     CSS3 anchor color in LAB space.
  4. Collapse each CSS3 name to one of **15 basic colors** via a
     hardcoded dict (`color_scheme` in that file). The full
     15-color vocabulary, taken from
     `language_generator/configs/metadata/metadata.json`
     `colors_used`: **maroon, brown, green, blue, pink, white,
     orange, black, olive, navy, purple, red, aqua, yellow,
     gray**.
  5. Take the 3 most common basic-color bins among all of the
     object's points. A bin only counts as "dominant" if
     `count >= 0.1 * total_points` (a literal **10% coverage
     floor**, `judge_color()` in the same file: `if
     index_pair[1] >= 0.1*len(indexes)`); slots that miss the
     floor are written as `'_'` (6 underscore placeholders: RGB
     triple, label, percentage, avg-dist).
  6. The stored representative RGB for a dominant bin is **not**
     the mean color of its points - it is the RGB of the single
     most-frequent individual CSS3 anchor color inside that bin.
  7. `object_color_scheme_average_distN` is the mean LAB distance
     from that bin's points to their individually matched CSS3
     anchor color (a same-bin tightness measure, not a distance to
     the bin's representative color).
- **Front-facing direction** (`object_front_heading`): only ever
  populated for **ScanNet** objects that have a Scan2CAD CAD
  alignment (`register_front_direction()` in
  `3d_data_preprocess/scannet/referit3d/referit3d/in_out/
  scan_2cad.py`, consumed in
  `3d_data_preprocess/scannet/scannet_preprocessing.py`: heading
  = `atan2` of the aligned CAD "front point" direction). For every
  other source it is hardcoded to `'_'` (N/A): confirmed literally
  in `3d_data_preprocess/arkit/arkit_data_generation.py` (`
  front_heading = ['_']`, unconditional) and in
  `3d_data_preprocess/unity/unity_preprocessing.py` (`
  object_line.append('_')` right after the bbox fields, no branch
  ever computes a real value). Matterport3D, HM3D and 3RScan's
  preprocessing scripts contain no "front" logic at all. **Net
  result: the Unity subset - the challenge's own sim engine - has
  zero populated front-facing-direction ground truth in this
  dataset**, confirmed by inspecting the actual sample CSV
  (`sample_data/Unity/loft/loft_object_result.csv`: every row's
  `object_front_heading` column is `_`).
- **Affordances**: a per-NYUv2-class (not per-instance) affordance
  word list is merged into every object node in the scene-graph
  JSON, sourced from
  `scene_graph/NYUv2_ChatGPT_Affordances.csv` (ChatGPT-generated
  per class). Not spotlighted in the paper/README table but
  present in the shipped scene graph - a free extra attribute.

## Spatial relations (exact definitions + thresholds)

All thresholds below are the argparse defaults in
`scene_graph/generate_scene_info.py` (`csv_to_json` ->
`compute_spatial_relationships`), which is what the released
dataset was actually built with (no separate config file
overrides these for the main release).

- **above** (`relate_above`): target counts as above anchor iff
  `anchor.max_z + on_thres <= target.min_z` **and** the 2D
  XY-plane intersection-over-min of their oriented footprints
  (`calculate_iom_poly`, i.e. `intersection_area /
  min(area_a, area_b)`) is `> vertical_iom`.
  Defaults: `on_thres = 0.01` (m), `vertical_iom = 0.5`.
- **below** (`relate_below`): target below anchor iff (target's
  `max_z <= anchor.min_z` and footprint IoM `> vertical_iom`, and
  target's class is not in a small `SUPPORTING_STRUCTURES`
  blacklist) **or** (target's `min_z <= anchor.min_z +
  under_thres` and target's `max_z <= anchor.max_z` and footprint
  IoM `> vertical_iom`, gated to only fire when the **anchor's**
  class is in `UNDER_RELATION`). `under_thres = 0.01` (m).
  `UNDER_RELATION` (classes with an "under-space", from
  `scene_graph/special_relation_classes.py`): cabinet, counter,
  table, desk, stool, shelf, drawer, dresser, bed, bookshelf, tv
  stand, bench, chest, piano bench, bar, night stand, coffee
  table.
- **on** (`relate_on`): target's `min_z` within
  `[anchor.min_z + under_thres, anchor.max_z + on_thres]`, footprint
  IoM `> vertical_iom`, anchor's XY footprint area `>` target's,
  anchor not a `VERTICAL_STRUCTURES` class (wall, door, door way,
  door frame, garage door), target not in `STRUCTURES_BLACKLIST`
  (wall, floor, ceiling, door, door way, door frame, garage
  door). Extra gate for `IN_ON_RELATION` classes (currently only
  **cabinet**, id 3): counts as "on" only if target's `max_z` is
  within the **top `in_thres` fraction (0.1 = 10%)** of the
  anchor's height band - i.e. resting near the top of a cabinet is
  "on", otherwise it is routed to "in" (see next).
- **in** (`relate_in`): target's center inside anchor's oriented
  bbox (signed-distance face-plane test, `is_inside_bbox`), anchor
  strictly larger than target on every axis, and target's z-range
  strictly inside anchor's z-range. Restricted to anchor classes in
  `IN_RELATION`: dishwasher, garbage bin, microwave, refrigerator,
  sink, box, bathtub, container, night stand, dresser, drawer,
  oven, toaster oven, washing machine, clothing dryer, clothing
  washer - or `IN_ON_RELATION` (cabinet), with the complementary
  gate to "on" above: an item in a cabinet only counts as "in" if
  its `max_z` is **not** within the top 10% (`in_thres = 0.1`) of
  the cabinet's height.
- **near** (`relate_near`): primary test is
  `euclidean(center_a, center_b) < near_thres * region_volume`,
  where `region_volume` is the region's bbox `xlen*ylen*zlen` (a
  **volume in m^3, not a length** - this is what the code
  literally computes, flagged as an odd-unit heuristic, not a
  transcription error) and `near_thres = 0.01`. A secondary
  corner-to-corner fallback exists in source (checks the minimum
  pairwise distance between the two objects' 8 bbox corners
  against the same threshold) but is **effectively dead code**: it
  compares a plain Python list to a float
  (`np.any(dists < near_thres * region_size) > 1`, where `dists`
  is built with `+=` on a list, never cast to `np.array`) - this
  was reproduced in isolation and raises `TypeError: '<' not
  supported between instances of 'list' and 'float'`. Since the
  full dataset generation run completed, this branch could not
  have executed for any object pair in the release; **the shipped
  "near" relation is therefore decided purely by the
  center-distance-vs-`0.01*region_volume` test**, not the
  corner-based fallback the code appears to intend.
- **closest / farthest** (`relate_ordered`): for each anchor
  object, objects are grouped by exact NYU class match and ranked
  by straight-line center-to-center distance. The **top 3** in each
  direction are kept (closest / second_closest / third_closest,
  and the farthest triple symmetrically) - not just rank 1.
  **Tie-break / ambiguity gate**: a class's closest-triple (or
  farthest-triple) is only emitted if, for every consecutive pair
  in the ranked triple, `abs(dist[i+1] - dist[i]) * avg_length >
  ordered_thres` (default `0.2`), where `avg_length` is the mean of
  each candidate's bbox length projected onto the anchor-direction
  vector. In other words: **ambiguous rankings are suppressed
  entirely rather than tie-broken** - if two same-class candidates
  are too close together relative to their own size, no
  closest/farthest statement is generated for that anchor+class at
  all.
- **between** (`relate_between`, ternary): for object *i* (the
  target) and every ordered pair of other objects *(j, k)*, build a
  local 2D frame whose x-axis is the direction from *j*'s to *k*'s
  center; *i* is "between" iff its center's projected coordinate
  falls strictly between *j*'s and *k*'s (checked independently in
  the XY-plane and, symmetrically, in Z - so a vertically-stacked
  "between two shelves" case is also caught). Additional gates,
  computed on signed 1D "intersection-over-min" values along that
  axis (negative when there is a gap, i.e. a size-normalized
  separation, not a raw-meter distance):
  - `overlap_thres = 0.3` - target's footprint must not overlap
    either flanking object by more than this on the between-axis.
  - `symmetry_thres = 0.5` - the two 1D-IoM-vs-each-flank values
    must not differ by more than this (keeps the target roughly
    centered rather than hugging one side).
  - `distance_thres = 1` - caps how separated the target can be
    from either flank (again in this size-normalized unit, not
    meters).
  - A separate, real 2D-IoM check (`between_iom = 0.5`) on the
    dimension-reduced projected boxes must also pass for both
    flanks.
  - Ranking/dedup: surviving pairs are sorted by the combined
    (more-negative-is-better) separation score, then **every other
    entry is kept via `[::2]` slicing** - a parity-based dedup
    heuristic to collapse each `(j, k)`/`(k, j)` mirror pair, not
    an exact-duplicate filter.
  - `anchor_size_thres` is accepted as an argparse argument
    (default `1.5`) but is **never referenced inside
    `relate_between`'s body** - it is a vestigial/no-op parameter
    in this release; do not spend effort trying to replicate an
    "anchor size" effect on `between` from this version of the
    code.
- General cross-relation filter (per README, and consistent with
  the IoM-based gates above): relation candidates are dropped if
  the target/anchor bounding boxes significantly overlap or one
  encloses the other.
- Two additional relations exist in the scene-graph JSON but are
  **not** part of the 8 documented in the paper's Table I and have
  no matching language-template file: `hanging_on` (objects
  floating above another surface within `hanging_thres_h = 0.01`
  horizontal footprint-distance and `hanging_thres_v = 0.5`
  vertical clearance, and not already "on"/"in" anything else) and
  a commented-out `beside`. Treat these as scene-graph-internal
  only, not part of the question-generation vocabulary.

## Referring-statement generation

- Three generator types (`language_generator/
  relationship_classes/`): `Binary_Relation` (above, below, near,
  on, in), `Ternary_Relation` (between), `Ordered_Relation`
  (closest/farthest, first/second/third rank).
- Per-relation template + synonym configs live under
  `language_generator/configs/relationship_configs/*.json`.
  Verbatim examples:
  - near: `"the %target_size%%target_color%%target% that
    %target_verb% %relation% the
    %other%%anchor_size%%anchor_color%%anchor%"`, synonyms `near,
    next to, close to, adjacent to, beside`.
  - between: `"the %target_size%%target_color%%target% that
    %target_verb% %relation% the %other1%%anchor1_size%
    %anchor1_color%%anchor1% and the
    %other2%%anchor2_size%%anchor2_color%%anchor2%"`, synonyms
    `between, in between, in the middle of`.
  - closest (first): synonyms `closest to, nearest to`.
  - farthest (first): synonyms `farthest from, most distant from`.
  - on: synonym is just `on`. in: synonyms `in, inside, within`.
    above: synonyms `above, over`.
- **Disambiguation cascade** (`ObjectFilter.filter_objects` /
  `filter_targets_and_anchors` in
  `language_generator/object_filtering/ObjectFilter.py`), applied
  to both the target and, first, the anchor:
  1. If the object has a dominant color (from its own top-3, not
     "N/A") whose coverage `> 0.25` (25%) and **no** other
     same-class distractor in the region shares that color label,
     that color alone disambiguates it - stop here.
  2. Else, rank same-class candidates by `largest_face_area`
     (largest of the 3 OBB face areas). If the target is the
     smallest and the next-smallest is `> 1.2x` its area, qualify
     with `"small "`; if it is the largest and it is itself `>
     1.2x` the next-largest, qualify with `"big "`.
  3. Else, repeat the size check **restricted to same-class
     objects sharing one of the target's dominant colors (still
     gated at the 25% coverage floor)** - producing a compound
     `"<color> <small|big> <class>"` qualifier.
  4. If none of these uniquely resolves the object, the caller
     discards the statement entirely (returns empty
     color/size lists, which the relation generators treat as
     "do not emit"). The **anchor** must already be resolvable this
     same way before the target is even attempted - an
     unresolvable anchor kills the statement regardless of the
     target.
- This realizes the paper's stated 3 properties: **view-independent**
  (relations never reference viewer perspective), **unique** (only
  one object in the region satisfies the full generated
  description), **minimal** (attributes are added in the fixed
  order above - relation+class first, then color, then size, then
  color+size - and only as far as needed to reach uniqueness).
- A parallel `get_false_statements[_ternary]` path in the same file
  generates hard-negative statements by substituting a color/class
  that is verified absent from every true occurrence of that
  (relation, target/anchor-class) slot in the region - i.e.
  distractor colors/classes are drawn from the complement of "what
  is actually true anywhere in this region for this slot," not
  arbitrary.
- Statement-type counts (README table, note types are not mutually
  exclusive): Above 47,208; Below 86,632; Closest 3,060,074;
  Farthest 4,590,111; Between 249,615; Near 1,655,185; In 11,157;
  On 25,915; Mentions color 3,485,373; Mentions size 2,114,500;
  **Total 9,696,079** (with relation synonyms counted separately;
  ~9.6M+ unique statements without synonyms, per README prose).

## File formats and Unity subset

Per-scene file tree (README, `<scene_name>/` under each dataset
folder): `<scene>_pc_result.ply` (colored point cloud),
`<scene>_object_split.npy` / `<scene>_region_split.npy` (id +
ending-index arrays to `numpy.split` the ply into per-object /
per-region point sets - points are pre-sorted by region then by
object id, with `-1` ids marking unlabeled points),
`<scene>_object_result.csv` / `<scene>_region_result.csv` (the flat
attribute tables described above; `OBJECT_HEADER` /
`REGION_HEADER` in `3d_data_preprocess/utils/headers.py`),
`<scene>_scene_graph.json`, `<scene>_referential_statements.json`.

- **Scene-graph JSON shape** (built by
  `scene_graph/generate_scene_info.py` `csv_to_json`):
  `{"scene_name": str, "regions": {region_id: {"region_id",
  "region_name", "region_bbox": [8 corner xyz triples], "objects":
  [{object_id, raw_label, nyu_id, nyu40_id, nyu_label, nyu40_label,
  color_vals (3x[r,g,b]), color_labels (3x str or "N/A"),
  color_percentages (3x), bbox (8 corners), center [x,y,z], volume,
  size [x,y,z], affordances [list of strings]}], "relationships":
  {above, below, closest, second_closest, third_closest, farthest,
  second_farthest, third_farthest, between, near, in, on,
  hanging_on: each a dict keyed by `object_id` -> list of related
  object id(s), or `[id1, id2]` pairs for between/hanging_on-style
  entries}}}}`.
- **Referential-statements JSON shape** (confirmed against the
  shipped Unity sample,
  `sample_data/Unity/loft/loft_referential_statements.json`): top
  level `{"scene_name", "regions": {region_id: {<full sentence
  text>: [{target_index, target_class, target_position [x,y,z],
  target_colors [3, may include "N/A"], target_size (bbox volume),
  target_color_used, target_size_used, distractor_ids, relation,
  relation_type, anchors: {anchor_1: {index, class, position,
  color, size, color_used, size_used}, anchor_2: {...only for
  ternary}}}]}}}`. Real example pulled from that file: the key
  `"the blanket that is between the table and the wardrobe"` maps
  to a one-element list with `relation: "between"`,
  `relation_type: "ternary"`, `target_colors: ["brown", "N/A",
  "N/A"]`, and two populated `anchors`.
- **Unity subset specifics**:
  - 15 released scenes + 3 held back "for the challenge," 46
    regions total (README). The public repo currently ships **17**
    `Categories_<scene>.csv` label-mapping files under
    `3d_data_preprocess/unity/labels/`: arabic_room, chinese_room,
    home_building_1, home_building_2, hotel_room_1, hotel_room_2,
    japanese_room, livingroom_1-4, loft, office_1, office_2,
    office_building_1, office_building_2, studio. That is neither
    15 nor 18 - **unreconciled discrepancy**, flagged rather than
    guessed at; would need a spot-check against an actual
    downloaded Unity zip (not done in this pass) to resolve.
  - Unity meshes are `.fbx`, parsed with the Autodesk FBX SDK
    Python bindings (`import fbx` in
    `3d_data_preprocess/unity/unity_preprocessing.py`). Region
    boundaries come from a **hand-authored**
    `<scene>_region.json` (per-region centroid, dimensions,
    z-rotation) rather than being algorithmically segmented - code
    comments flag "a rotation difference, need to switch axis"
    when reading these, i.e. an axis-convention gotcha between the
    region JSON and the mesh/point-cloud frame.
  - Point clouds are sampled uniformly from the UV-mapped mesh
    surfaces (not scanned); per-scene sample count is proportional
    to **the number of objects in the scene** (README) - unlike
    HM3D/3RScan, which sample proportional to total mesh surface
    area divided by `2e-4`.
  - `object_front_heading` is unconditionally `_` (N/A) for every
    Unity object - confirmed both in code
    (`unity_preprocessing.py`) and in the shipped sample CSV. The
    challenge's own sim engine therefore carries **no
    front-facing-direction ground truth** in this dataset.

## Takeaways for our 2026 module

- If our test-time scene graph should speak the same relation
  vocabulary the questions were generated from, replicate the
  **gates**, not just the relation names: `on`/`in`/`above`/`below`
  need a Z-band test (`on_thres`/`under_thres` = 0.01 m) combined
  with a 2D footprint intersection-over-min `> 0.5`; `near` is a
  center-distance test scaled by **region volume** (`0.01 *
  xlen*ylen*zlen`), not a fixed radius - a fixed-radius "near"
  will disagree with how the source questions were generated,
  especially between small and large rooms.
- `closest`/`farthest` are **not** single nearest-neighbor
  relations in the source data - they are top-3 within class, and
  crucially the source generator **suppresses** the statement
  entirely when the ranking is size-relative-ambiguous
  (`ordered_thres = 0.2` gate). A grounding module that always
  answers "the closest X" even when two X's are near-tied is
  answering a class of question the dataset's own generator would
  have refused to ask - worth replicating that suppression logic
  if we generate our own candidate statements for self-check.
- `between` is genuinely ternary and its gates operate on
  size-normalized separation, not raw distance - a naive "target's
  center x is between anchor1.x and anchor2.x" check will produce
  many more (and differently-shaped) matches than the source
  generator's `overlap_thres`/`symmetry_thres`/`distance_thres`
  combination. If we need to answer "between" questions
  robustly, reimplement the projected-frame + IoM-gate approach,
  not a bare linear-interpolation test.
- Colors: the question vocabulary is a **closed 15-color set**
  (maroon, brown, green, blue, pink, white, orange, black, olive,
  navy, purple, red, aqua, yellow, gray), each requiring `>= 10%`
  point coverage to be "dominant," with up to 3 dominant colors
  per object. A perception pipeline only needs to classify into
  these 15 bins (e.g. via LAB-space nearest-color or a small
  learned classifier), not general free-form color naming, to
  match how "blue chair"-style questions were generated.
- Disambiguation cascade (relation -> unique color at >25%
  coverage -> size outlier at 1.2x gap -> color+size) is a
  reusable heuristic for **our own** object descriptions: if we
  need to generate a referring phrase for an object we've
  detected, this exact order (cheapest/most-natural attribute
  first) matches what the question set already assumes disambig-
  uation looks like. It is also a reading aid: a challenge
  question that already carries a color or size qualifier is a
  signal that the source scene had multiple same-class
  distractors, i.e. don't assume the qualifier is decorative.
- Do not build any reasoning that depends on a canonical
  "front-facing direction" for Unity-sourced objects - that
  attribute does not exist for the challenge's own sim engine in
  this dataset. If the challenge needs facing-relative reasoning
  (e.g. "the chair facing the window"), that ground truth must
  come from somewhere other than VLA-3D's Unity subset.
- Two implementation details in the reference generator are
  effectively dead/no-op in the shipped release: the `near`
  relation's corner-distance fallback (raises a `TypeError` if
  ever reached - verified by isolated repro) and `between`'s
  `anchor_size_thres` argument (never read in the function body).
  Don't spend effort trying to reverse-engineer intended behavior
  from these - the actual dataset was generated without them
  ever firing.
- The per-class affordance list
  (`scene_graph/NYUv2_ChatGPT_Affordances.csv`) is a free extra
  signal, present in the scene graph but not discussed in the
  paper - potentially useful for instruction-following queries
  that reference object function rather than identity (e.g. "put
  it somewhere you can throw it away").
- Follow-up to check later: **IRef-VLA**
  (https://github.com/HaochenZ11/IRef-VLA , arXiv 2503.17406) is
  described by the VLA-3D README as a filtered/extended successor
  with "misaligned grounding statements" - if the challenge's
  question generation moved to reference that dataset instead,
  the relation/threshold spec above could be stale. Not checked in
  this pass.

## Sources

- Paper (abstract, authors, venue, version history):
  https://arxiv.org/abs/2411.03540
- Paper full text (HTML, used only for cross-checking prose
  claims not verifiable in code, e.g. Unity annotator count):
  https://arxiv.org/html/2411.03540v1
- Main repo (all code/threshold citations above; cloned shallow at
  commit `a0c023c9662bd875f0b3aeff4ff60adc1991e449`,
  2026-03-02): https://github.com/HaochenZ11/VLA-3D
  - `README.md` - dataset scale, per-source scene counts, file
    tree, statement-type counts, dominant-color 10%/top-3 summary.
  - `scene_graph/generate_scene_info.py` - all relation thresholds
    and gating logic (`relate_above`, `relate_below`, `relate_on`,
    `relate_in`, `relate_near`, `relate_ordered`, `relate_between`,
    `relate_hanging_on`, and the argparse defaults).
  - `scene_graph/bbox_utils.py` - IoM/IoU helpers, oriented-bbox
    construction, `is_inside_bbox`.
  - `scene_graph/special_relation_classes.py` - the
    `IN_RELATION` / `ON_RELATION` / `IN_ON_RELATION` /
    `UNDER_RELATION` / `STRUCTURES_BLACKLIST` /
    `SUPPORTING_STRUCTURES` class lists.
  - `scene_graph/colors.py` - CSV-column-to-object-dict color
    parsing used downstream by the language generator.
  - `3d_data_preprocess/utils/dominant_colors_new_lab.py` - the
    LAB conversion, CSS3-anchor KD-tree, 15-basic-color mapping,
    and the 10% dominance floor.
  - `3d_data_preprocess/utils/headers.py` - exact CSV column
    order for objects and regions.
  - `3d_data_preprocess/unity/unity_preprocessing.py` - Unity FBX
    parsing, hand-authored region JSON, object-count-proportional
    point sampling, and the hardcoded `object_front_heading = '_'`.
  - `3d_data_preprocess/arkit/arkit_data_generation.py` and
    `3d_data_preprocess/scannet/scannet_preprocessing.py` (plus
    `.../referit3d/referit3d/in_out/scan_2cad.py`) - confirms
    `object_front_heading` is ScanNet/Scan2CAD-only.
  - `language_generator/relationship_classes/*.py`,
    `language_generator/object_filtering/ObjectFilter.py`,
    `language_generator/Statement.py` - generator classes,
    disambiguation cascade, false-statement generation, statement
    object schema.
  - `language_generator/configs/relationship_configs/*.json`,
    `language_generator/configs/metadata/metadata.json` - verbatim
    templates, synonyms, and the 15-color vocabulary list.
  - `sample_data/Unity/loft/*` - real shipped Unity sample used to
    confirm the scene-graph/statement JSON shapes and the
    all-`_` `object_front_heading` column.
- IRef-VLA follow-up (filtered/extended successor, not reviewed
  beyond this pointer): https://github.com/HaochenZ11/IRef-VLA ,
  https://huggingface.co/papers/2503.17406
- Cross-reference: this repo's own shallower pass on VLA-3D,
  `docs/prior_art/README.md` section 2 ("VLA-3D dataset takeaways").
