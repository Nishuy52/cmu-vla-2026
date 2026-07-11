# VLA-3D Dataset Notes — CMU VLN Challenge 2026

**Date:** 11 Jul 2026
**Scope:** Everything needed to decide whether/how to pull VLA-3D's Unity subset for the 15
challenge training scenes, plus the exact per-scene file schema and the local trajectory PLY
format. Sources: [VLA-3D GitHub repo](https://github.com/HaochenZ11/VLA-3D) (README, LICENSE,
`download_dataset.py`, `sample_data/Unity/loft/*`), [paper arXiv:2411.03540](https://arxiv.org/abs/2411.03540),
[follow-up repo IRef-VLA](https://github.com/HaochenZ11/IRef-VLA), and the local upstream clone at
`upstream/CMU-VLN-Challenge-2026/`.

---

## 0. Verdict (read this first)

- **Hosting:** not HuggingFace, not Google Drive, not generic AWS S3 — it's **CMU AirLab's own
  S3-compatible (Ceph/OpenStack Swift) object store**, anonymous/unsigned access via `boto3`.
- **Per-source download is possible and is the only granularity offered.** There is no per-scene
  download — the `--subset Unity` flag pulls one file, `Unity.zip`, containing **all 18** Unity
  scenes (15 training + 3 held-out test), not selectable further. You cannot fetch just
  `loft.zip`.
- **Size of `Unity.zip` confirmed by direct HTTP HEAD request:** `Content-Length: 2,093,930,095`
  bytes = **~1.95 GiB (2.09 GB)**. That is the whole Unity subset, all 18 scenes together — small
  by dataset standards and one file, so it should survive a throttled VPN as a single resumable-ish
  download (S3 GET, not resumable by default in the provided script, but `Range` requests work
  against the endpoint since `accept-ranges: bytes` was returned).
- **Scene-name match to the challenge is exact.** The GitHub sample data ships a full worked
  example for scene `loft` under `sample_data/Unity/loft/`, and its `region_result.csv` /
  `object_result.csv` content lines up with what the challenge's own `loft` scene would produce.
  VLA-3D's Unity folder names are the same flat `<scene_name>/` scheme the challenge repo uses.
  No renaming/remapping layer exists between the two.
- **Recommended download command** (see §1 for full context): only the Unity subset is needed —

  ```bash
  pip install boto3 tqdm
  python download_dataset.py --download_path vla3d_unity --subset Unity
  ```

  This fetches `vla3d_unity/Unity.zip` (~2.1 GB), which unzips into per-scene folders including
  all 15 training scenes named identically to `upstream/CMU-VLN-Challenge-2026/questions/*`.

---

## 1. Download mechanics

### Hosting

Confirmed by reading `download_dataset.py` directly (raw file, GitHub main branch):

```python
BUCKET = "vla"
ENDPOINT = "https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330"
```

This is CMU's own Ceph Object Gateway exposed over an S3-compatible Swift API
(`server: Ceph Object Gateway (tentacle)` in the HTTP response headers). Access uses `boto3` with
`UNSIGNED` config — no AWS account, credentials, or sign-up needed. Not HuggingFace (searched;
no VLA-3D listing exists there), not Google Drive/S3-proper.

### Download script and options

Repo root: `download_dataset.py`. Usage from the README:

```bash
pip install boto3 tqdm
python download_dataset.py --download_path full_dataset
```

Arguments:
- `--download_path` — output folder (defaults to `VLA-3D_dataset`)
- `--subset` — one of `Matterport` / `Scannet` / `HM3D` / `Unity` / `ARKitScenes` / `3RScan`
  (case-insensitive, matched via a lowercase dict lookup in the script). Omit for the full 7,635-scene
  dataset (not needed for our purposes — real-scan sources like ARKitScenes alone are 4,494 scenes,
  multi-tens-of-GB; do not download those).

Internally the script maps subset name to filename via a fixed dict:
```python
source_files = {"matterport":"Matterport.zip", "scannet":"Scannet.zip", "hm3d":"HM3D.zip",
                 "unity":"Unity.zip", "arkitscenes":"ARKitScenes.zip", "3rscan":"3RScan.zip"}
```
and does `client.get_object(Bucket="vla", Key="Unity.zip")`, streaming the body to disk in 1 MB
chunks with a `tqdm` progress bar. **Granularity is exactly one zip per source dataset — no
per-scene, per-region, or byte-range slicing is exposed in the script itself.** You could
in principle issue your own `Range` GET against the same S3 key to grab a partial file, but the
zip's central directory is at the end and scene files aren't laid out for trivial partial
extraction, so this isn't worth pursuing given the file is only ~2 GB.

### Unity.zip size — verified directly

```
$ curl -sI https://airlab-cloud.andrew.cmu.edu:8080/swift/v1/AUTH_ac8533a83cff4d48bc8c608ad222d330/vla/Unity.zip
HTTP/1.1 200 OK
content-length: 2093930095
accept-ranges: bytes
last-modified: Wed, 19 Nov 2025 20:04:38 GMT
```

2,093,930,095 bytes ≈ **1.95 GiB**. This is the entire Unity source subset — 18 scenes (15
training + 3 held out for the challenge's own test evaluation, per the VLA-3D README: "15 scenes +
3 scenes omitted for the challenge"). There is no way to shrink this further via the official
script; the whole zip must be pulled to get any of the 15 training scenes' processed data. Given
the ~2 GB total, this is **not** a multi-GB-per-scene problem — it is one ~2 GB download for all
15 scenes combined, which is the good case.

For comparison (do not download): ARKitScenes = 4,494 scenes, 3RScan = 1,381 scenes, ScanNet =
1,513 scenes — all substantially larger and irrelevant to this challenge.

---

## 2. Per-scene file format

Confirmed against both the README's documented pipeline output and a real worked example
(`sample_data/Unity/loft/`, fetched directly).

### Documented file set (from README "Dataset Format" section, verbatim structure)

```
<dataset_folder>/
 -- <scene_name>/
    -- <scene_name>_pc_result.ply           Processed point cloud of entire scene
    -- <scene_name>_object_split.npy        Object IDs + split indices for the .ply file
    -- <scene_name>_region_split.npy        Region IDs + split indices for the .ply file
    -- <scene_name>_object_result.csv       Per-object: id, class labels, bbox, dominant colors
    -- <scene_name>_region_result.csv       Per-region: id, region name, bbox
    -- <scene_name>_scene_graph.json        Object relations within each region/room
    -- <scene_name>_referential_statements.json   Generated language statements
```

**Note on sample-data naming drift:** the actual files shipped under `sample_data/Unity/loft/` in
the repo are `loft_object_result.csv`, `loft_object_split.npy`, `loft_referential_statements.json`,
`loft_region_result.csv`, `loft_region_split.npy`, `loft_scene_graph.json` — matching the
documented set — **plus two extra files not in the documented list**: `loft_objects.json` and
`object_list.txt` (undocumented, likely leftover/intermediate artifacts from an older pipeline
version bundled into the sample only). The full downloaded `Unity.zip` should be checked for
whether these extras are present per-scene; treat the seven documented files as authoritative and
the two extras as bonus/ignorable if present.

There is no separate free-space annotation file in the file list — the README describes
free-space generation as a processing step ("each scan was also processed to generate the
horizontally traversable free space... chunked into sub-regions... spatial relations generated")
but does not name a distinct output file for it. Free-space sub-regions appear to be folded into
the scene graph / referential statements as objects rather than a standalone artifact. Confirm
this once the real zip is unpacked — the README is not fully explicit here.

### Point cloud (`<scene>_pc_result.ply`)

ASCII or binary PLY (format not stated) storing `x,y,z` + RGB per point, **no per-point object/region
ID** — instead points are pre-sorted first by region ID then by object ID within each region, and
the two `.npy` split-index files let you recover per-object/per-region point ranges via
`numpy.split`. For Unity specifically (unlike Matterport/ScanNet/ARKitScenes, which have colors
pre-baked into native `.ply` meshes), the source mesh is `.fbx` with UV-mapped textures, so VLA-3D
uniformly samples points from the mesh and bakes texture colors onto them; sample density for
Unity scenes is proportional to object count in the scene (not surface area, unlike HM3D/3RScan).

### Object CSV (`<scene>_object_result.csv`) — real header + 3 rows from `loft`

```
object_id,region_id,raw_label,nyu_id,nyu40_id,nyu_label,nyu40_label,object_bbox_cx,object_bbox_cy,object_bbox_cz,object_bbox_xlength,object_bbox_ylength,object_bbox_zlength,object_bbox_heading,object_front_heading,object_color_r1,object_color_g1,object_color_b1,object_color_scheme1,object_color_scheme_percentage1,object_color_scheme_average_dist1,object_color_r2,object_color_g2,object_color_b2,object_color_scheme2,object_color_scheme_percentage2,object_color_scheme_average_dist2,object_color_r3,object_color_g3,object_color_b3,object_color_scheme3,object_color_scheme_percentage3,object_color_scheme_average_dist3

0,2,dvd,197,40,dvd,otherprop,5.907001458270205,-1.4560003448913281,0.3147824715963571,0.39537708756611956,0.22755075574315842,0.048072284481764926,-0.018558140622046932,_,47,79,79,gray,0.9513294276701216,26.619727476622305,_,_,_,_,_,_
1,1,door,28,8,door,door,7.856646697541533,-1.8823342034670352,1.077385822045167,0.16891928423855385,0.8900627127442524,2.193669894054672,1.5707963267948966,_,169,169,169,gray,0.9912509355563012,1.5844154583035204,_,_,_,_,_,_
```

Field meaning (per README):
- `object_id`, `region_id` — object's own id and the region it belongs to
- `raw_label` — original/native label string (e.g. `dvd`, `door`, `door frame`) — this is the
  field to match against challenge question nouns, not the NYU labels, since raw labels are
  closer to natural free-text ("bowl of apples", "potted bamboo", "sphere decoration" all appear
  in the `loft` sample)
- `nyu_id`, `nyu40_id`, `nyu_label`, `nyu40_label` — NYUv2 / NYU40 class index+name mapping
  (coarser, standardized categories — e.g. `dvd` maps to NYU40 `otherprop`)
- `object_bbox_c[xyz]`, `object_bbox_[xyz]length`, `object_bbox_heading` — oriented bounding box
  center, extents (length/width/height), and heading (yaw) angle
- `object_front_heading` — canonical "front" direction if the object has one (`_` = none, as seen
  for `dvd`/`door` above)
- `object_color_[rgb][1-3]`, `object_color_scheme[1-3]`, `object_color_scheme_percentage[1-3]`,
  `object_color_scheme_average_dist[1-3]` — up to 3 dominant colors per object (RGB triplet, one
  of 15 canonical color names, fraction of points assigned, and LAB-space distance-to-anchor
  quality metric); `_` fills unused slots when fewer than 3 dominant colors exist

### Region CSV (`<scene>_region_result.csv`) — full real content, `loft` scene

```
region_id,region_label,region_bbox_cx,region_bbox_cy,region_bbox_cz,region_bbox_xlength,region_bbox_ylength,region_bbox_zlength,region_bbox_heading
0,bedroom,0.10655447,-0.31046733,4.08751222,4.18013625,8.08272244,2.84436836,6.278373619262206
1,corridor,3.66720341,-2.97204946,1.39810654,2.23935906,11.17767733,2.7314005,1.574263880979313
2,livingroom,3.58524903,0.85110725,1.2813705,5.63582718,10.98320993,2.93367214,4.7122987205078894
```

So `loft` — a single "Unity scene" in challenge terms — actually decomposes into **3 VLA-3D
regions** (bedroom/corridor/livingroom) internally; this multi-region structure exists per Unity
scene generally (the README states the Unity source overall has 46 regions across its 18 scenes,
~2.5 regions/scene average). Region bounding boxes for Unity are axis-aligned, heading currently
unused (README says heading is 0 for all datasets at the region level, though the raw CSV values
above aren't exactly 0 — likely a coordinate-frame artifact; treat with mild caution and re-derive
if precision matters). Heading angle is in radians (`6.278...` ≈ 2π ≈ ~0 rad, consistent with the
README's claim).

### Scene graph (`<scene>_scene_graph.json`) — real excerpt, `loft`

```json
{
    "scene_name": "loft",
    "regions": {
        "0": {
            "region_id": "0",
            "region_name": "bedroom",
            "region_bbox": [ [x,y,z], [x,y,z], ... 8 corner points ... ],
            "objects": [
                {
                    "object_id": "5",
                    "raw_label": "lamp",
                    "nyu_id": "144",
                    "nyu40_id": "35",
                    "nyu_label": "lamp",
                    "nyu40_label": "lamp",
                    "color_vals": [ [r,g,b], [-1,-1,-1], [-1,-1,-1] ],
                    ...
                }
            ]
        }
    }
}
```

Keyed by region ID, each holding a full 8-corner 3D bounding-box polygon for the region plus a
nested `objects` list (same fields as the object CSV, restructured as JSON) — and, per the
README, inter-object relation edges (Above/Below/Closest/Farthest/Between/Near/In/On) computed
per-region. The relation table from the README:

| Relation | Definition | Synonyms | Notes |
|---|---|---|---|
| Above / Below | vertical order, no contact | Over / Under, Beneath, Underneath | |
| Closest / Farthest | nearest/farthest same-class object to an anchor | Nearest / Most distant from | inter-class |
| Between | target lies between two anchors | In the middle of, In-between | ternary |
| Near | within a distance threshold | Next to, Close to, Adjacent to, Beside | symmetric |
| In | target inside anchor | Inside, within | |
| On | vertical contact | On top of | |

All relations are view-independent (perspective-agnostic) and filtered to avoid heavily
overlapping/enclosing boxes.

### Referential statements (`<scene>_referential_statements.json`) — real excerpt, `loft`

```json
{
    "scene_name": "loft",
    "regions": {
        "0": {
            "the book that is above the big table": [
                {
                    "target_index": "72",
                    "target_class": "book",
                    "target_position": [-0.332, -3.376, 3.760],
                    "target_colors": ["gray", "N/A", "N/A"],
                    "target_size": 0.0015263447270627002,
                    "target_color_used": "",
                    "target_size_used": "",
                    "distractor_ids": ["58"],
                    "relation": "above",
                    "relation_type": "binary",
                    "anchors": {
                        "anchor_1": {
                            "index": "22",
                            "class": "table",
                            "position": [-0.469, -3.507, 3.329],
                            "color": ["gray", "N/A", "N/A"],
                            "size": 0.7859464934072631,
                            "color_used": "",
                            "size_used": "big "
                        }
                    }
                }
            ]
        }
    }
}
```

Top-level keys per region are the natural-language statement strings themselves, mapping to a
list of grounding-annotation dicts (usually one). Fields: `target_index`/`target_class`/
`target_position`/`target_colors`/`target_size` describe the referred object; `distractor_ids`
lists same-class objects the statement must disambiguate from; `relation`/`relation_type`
(`binary`/`ternary`/`ordered`) describe the spatial predicate; `anchors` holds one or more
reference objects (`anchor_1`, `anchor_2`, ... for `between`) with the same position/color/size
fields plus `color_used`/`size_used` (non-empty when that attribute was needed to disambiguate,
e.g. `"big "` above). Per-scene sizes are large — `loft`'s statement file alone is ~8.5 MB even
for a 3-region scene, since statements are generated exhaustively per relation type; expect
several MB per scene across the 15 training scenes, negligible next to the 2 GB point-cloud/zip
total.

Statement-type volume (whole VLA-3D dataset, all sources, from README table) for context on the
generation method (not per-scene, but shows base rates): Closest 3.06M, Farthest 4.59M, Near
1.66M, color/size-augmented majority of the ~9.7M total.

---

## 3. License

Repo root `LICENSE` file (`raw.githubusercontent.com/HaochenZ11/VLA-3D/main/LICENSE`), fetched
verbatim: **MIT License**, `Copyright (c) 2024 Haochen Z`. Standard MIT terms — permissive,
allows commercial/derivative use, no attribution requirement beyond retaining the notice in
copies of the software itself. No separate data-specific license or usage restriction found in
the repo; the MIT LICENSE file covers the repo (code + implicitly the released data artifacts,
per common practice, though the file itself only says "Software").

Follow-up dataset **IRef-VLA** (`github.com/HaochenZ11/IRef-VLA`, arXiv:2503.17406, ICRA 2025) is
explicitly "an extension on top of VLA-3D" — same six source datasets including the identical
Unity 15+3 scene split, same `--subset Unity` download mechanism and script, same documented file
schema, but with added "augmented false statements" (imperfect/ambiguous language) and a smaller
total statement count (4.7M vs 9.7M — the README says VLA-3D "has been filtered and extended with
misaligned grounding statements" in IRef-VLA, i.e. it's a refinement/superset in annotation
quality, not raw data volume). **Not needed for this challenge** — VLA-3D's plain
`referential_statements.json` is the right artifact for straightforward object-reference
grounding; IRef-VLA's ambiguity-augmented statements are for benchmarking robustness to
imperfect language, out of scope here. Worth knowing it exists in case object-grounding accuracy
becomes a bottleneck later.

---

## 4. Scene-name mapping and object-noun match to challenge questions

### Authoritative scene list (from local upstream clone, not the prompt's assumed list)

Read directly from `upstream/CMU-VLN-Challenge-2026/questions/questions.json` and cross-checked
against the folder listing under `questions/`. **The actual 15 training scenes are:**

```
arabic_room, chinese_room, home_building_1, home_building_2, hotel_room_1, hotel_room_2,
japanese_room, livingroom_1, livingroom_2, livingroom_3, livingroom_4, loft, office_1,
office_2, studio
```

This differs from the scene names assumed going in — there is **no** `home_building_3`,
`restaurant`, `seminar_room`, or `western_restaurant` in the actual challenge repo. The correct
set has `livingroom_3`, `livingroom_4`, `office_1`, `office_2`, and `studio` instead. Use this
list, not any list derived from memory, for all downstream tooling (file matching, scan scripts,
etc.).

### 1:1 folder-name match confirmed

The challenge's own README states plainly (Setting Up → Object-Referential Dataset section):
"[VLA-3D] includes the 15 training scenes in Unity." The VLA-3D README independently states the
Unity source has "15 scenes + 3 scenes omitted for the challenge," and the GitHub sample data
ships a full example under `sample_data/Unity/loft/` — `loft` being one of the 15 challenge scene
names verbatim. This is strong direct evidence VLA-3D's Unity scene folder names match the
challenge's scene names exactly (flat `<scene_name>/` layout on both sides, no renaming/mapping
layer). Recommend confirming the other 14 names appear identically once `Unity.zip` is actually
unzipped, but there is no reason from the docs to expect divergence — same organizers, same
scene set, explicitly cross-referenced in both READMEs.

### Object-noun match

Spot check: `loft` sample `raw_label` values include `sofa`, `pillow`, `table`, `lamp`, `vase`,
`tv`, `chair`, `cabinet`, `potted plant`, `fireplace`, `stairs` — and the challenge's `loft`
question set (`questions.json`) asks about exactly these: "How many black pillows are on the
sofa?", "the blue chair that is closest to the cup of coffee", "potted plant between a vase and
the cabinet with a TV on it", "near the fireplace, pass by the stairs... sphere decoration on the
cabinet" (and `sphere decoration` also literally appears in the `loft` object CSV as a
`raw_label`). **Match is exact at the free-text (`raw_label`) level for every noun checked.** Use
`raw_label`, not `nyu_label`/`nyu40_label`, when matching challenge question nouns to VLA-3D
objects — NYU labels are coarser (e.g. `dvd` → NYU40 `otherprop`) and would lose the specific
wording the questions use.

---

## 5. Trajectory ground-truth format (`trajectory_q4.ply` / `trajectory_q5.ply`)

Read directly and completely locally — no download needed, files already present under
`upstream/CMU-VLN-Challenge-2026/questions/<scene>/`. Verified on `arabic_room/trajectory_q4.ply`
(792 vertices) and `arabic_room/trajectory_q5.ply` (836 vertices); pattern holds across all 15
scene folders (each has exactly `trajectory_q4.ply` + `trajectory_q5.ply`, i.e. one target
trajectory per instruction-following question — scenes have 2 instruction-following questions
each per `questions.json`, hence q4/q5, presumably the 4th/5th question slot for each scene after
1 numerical + 2 object-reference).

### Exact structure (verbatim header + first/last data rows)

```
ply
format ascii 1.0
element vertex 792
property float x
property float y
property float z
end_header
-0.010495 -0.000302 0.75
-0.021435 -0.000778 0.75
...
2.48279 -1.22576 0.75
```

- Plain **ASCII PLY**, no faces/edges, vertex-only point list — this is a waypoint path, not a
  mesh.
- Exactly 3 float properties per vertex: `x y z`. No color, no normal, no extra per-vertex
  fields, no comment lines.
- **`z` is constant (`0.75`) across all 792 vertices** in the sample checked — confirmed by
  extracting the full z-column and finding a single unique value. This is a fixed-height 2D path
  (robot/camera reference height), not a 3D trajectory with elevation changes. Safe to treat as
  `(x, y)` waypoints with a fixed z when consuming these files; don't over-interpret the z-column
  as meaningful terrain-following data.
- Point ordering is sequential path order (monotonically progressing x/y between rows, consistent
  with a continuous walked/driven trajectory rather than a point cloud).
- File sizes are tiny (a few hundred vertices, well under 1 KB/vertex as ASCII) — no download or
  size concern at all; already local.

No PLY header comments, no format variation observed between the two files checked. Recommend
treating this as the fixed schema for all 15 × 2 = 30 trajectory files without needing to
re-verify each one, but a quick per-file vertex-count sanity check costs nothing if used
programmatically.

---

## 6. Open questions / things to verify once `Unity.zip` is actually unzipped

1. Whether the two undocumented sample files (`loft_objects.json`, `object_list.txt`) are present
   for every scene in the real zip, or were sample-only artifacts.
2. Whether there's a distinct free-space annotation file per scene (not called out in the
   documented 7-file list) or whether free-space sub-regions are only reachable via the scene
   graph/referential statements.
3. Confirm all 15 challenge scene names appear as top-level folders inside `Unity.zip` with no
   spelling/casing drift (spot-checked only `loft` via GitHub sample data so far).
4. PLY encoding (ascii vs binary_little_endian) for the *point cloud* files specifically — the
   trajectory PLYs are confirmed ASCII, but `<scene>_pc_result.ply` files were not directly
   inspected (would require the actual download) and could plausibly be binary given they contain
   full colored point clouds at real scene scale.
