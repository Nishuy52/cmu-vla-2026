> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# Dev-kit fork network - competing-team snapshot

Point-in-time snapshot, captured 2026-07-11 via `gh api` against
GitHub's fork/branch/commit endpoints. Fork activity changes
constantly - re-run the survey before relying on this for a
go/no-go call; treat everything below as "as of 2026-07-11," not a
live view.

## Scope

Checked the fork networks of all three repos in
`docs/challenge_brief.md`'s Sources list:

- Dev kit: `Yuxin916/CMU-VLN-Challenge-2026` (renamed from
  `CMU-VLA-Challenge-2026` - GitHub silently redirects the old
  slug; update references if the rename causes confusion).
- Dataset: `HaochenZ11/VLA-3D`
- Prior-year base system: `jizhang-cmu/cmu_vla_challenge_unity`

Only the dev kit's fork network carries this-year signal. VLA-3D
(7 forks) and the prior-year base repo (5 forks) both have forks,
but every one of them was last pushed in 2024 or early 2026 before
this challenge cycle started - general dataset/research forks, not
competing teams. Not covered further below.

## Dev kit forks: 8 total, 2 active

6 of 8 forks are untouched clones (0 commits ahead of
`Yuxin916/CMU-VLN-Challenge-2026:main`, `main` branch only, no
other activity). The other 2 have real work, both converging on the
same "scene graph + VLM" shape as our own `docs/prior_art/README.md`
baseline pick.

### `jainanshu0912/CMU-VLN-Challenge-2026`

Two branches, `Cursor`-assisted commits, last push 2026-07-10.

- `track-a-vlm-pipeline`: `ai_module/src/vlm_pipeline` - zero-shot
  find/count over a **static scene graph built from VLA-3D data**,
  with a pluggable VLM backend layer
  (`vlm_pipeline/vlm_backends/`: Claude, Gemini, OpenAI, Ollama).
  The multi-backend hedge mirrors our own open question in
  `docs/challenge_brief.md` about which API is legal/available at
  eval time.
- `track-b-vlm-pipeline-live`: adds
  `ai_module/src/vlm_pipeline_live` - GroundingDINO + LiDAR fusion
  for live/manual-teleop mapping, layered on top of Track A rather
  than replacing it.

### `kante2/CMU-VLN-Challenge-2026`

Two branches, commit messages partly in Korean, last push
2026-07-09.

- `feature/eval`: local evaluation harness, plus a `tmah_vlm`
  skeleton (sensor subscriptions, question dispatcher,
  GroundingDINO perception stubs).
- `feature/graph`: extends into `ai_module/src/tmah_vlm/tmah_vlm/
  bbox3d/` (3D box estimation) and `.../graph/` (scene-graph nodes/
  edges/runtime), with a commit explicitly titled "Add scene graph
  and SORT3D visualization support" - direct evidence a competing
  team is integrating SORT3D (already our top prior-art pick), not
  just citing it.

## Read for our approach

No fork is meaningfully ahead in a way that changes the
conclusions in `docs/prior_art/README.md`. Both active teams are
building the same "detect -> object-centric scene graph -> VLM
reasons over it" pipeline already identified as the likely-strong
baseline in `docs/prior_art/README.md` - validating, not alarming. Neither
has visible instruction-following (waypoint/navigation) work yet;
both are still on the perception/grounding side.

## How to re-run this survey

```
gh api repos/Yuxin916/CMU-VLN-Challenge-2026/forks --paginate \
  --jq '.[] | [.full_name, .pushed_at] | @tsv'

# per fork, ahead/behind vs upstream main:
gh api repos/<owner>/CMU-VLN-Challenge-2026/compare/Yuxin916:main...<owner>:main \
  --jq '{ahead_by, behind_by}'

# per fork, non-main branches (where the real work usually lives):
gh api repos/<owner>/CMU-VLN-Challenge-2026/branches --jq '.[].name'
```
