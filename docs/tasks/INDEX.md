# Task Index

- **T1** cmu-vln-preliminary-discussion — prep material (problem
  statement, examples, competition-process specifics) for the
  team's preliminary discussion on the CMU VLN Challenge.
- **T2** 2025-results-research — identify CMU VLA Challenge 2025
  top-1/top-2 teams and their methods; append to
  docs/prior-art.md section 1.
- **T3** merge-into-team-repo — fold the parallel research
  stream into the team repo (github.com/Nishuy52/cmu-vla-2026)
  on a branch + PR; append-first, provenance-tagged.
- **T4** implementation-gaps — top-5 discrepancies (missing /
  contradictory) between the implementation and the imported
  research findings; ranked, cited, one candidate falsified.
- **T5** colored-cloud-tool — offline colored point-cloud
  reconstruction debug tool (`tools/`): lidar → pano color →
  voxel downsample → PLY; validated on real jingfan data.
- **T6** live-colored-map — incremental colored voxel map
  (`core/perception/colored_map.py`) + debug-gated RViz
  PointCloud2 publisher: live "robot inside the colored map"
  view; debug layer only, scored path unchanged.
