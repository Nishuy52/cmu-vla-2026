# Questions

Running FAQ for this task. Newest first.

---

## Q1 (2026-07-11): What is "GT semantics"?

### Answer

Ground-truth semantics: object knowledge the simulator hands
over directly instead of the robot perceiving it. In 2025 the
dev kit broadcast `/object_markers` - labeled 3D bounding boxes
(class, center, size) for objects within ~2 m of the robot,
straight from the sim's per-scene object list. This deletes the
hard perception problem (open-vocab detection, 3D localization,
instance merging); only color was missing (CopyPasta recovered
it by sampling camera pixels at projected marker centroids).
Legal in 2025 but "scored differently" per the 2025 README.
Most 2025 entries depended on it: Anand 2024 (voxel map built
purely from markers, OwlViT as text encoder only), ReasonX
(query-to-marker-name matching), CopyPasta (T1/T2), URL-KAIST
(GT fused at confidence 1.0 over its own YOLO). The 2026
contract removes the topic, so those pipelines do not port and
open-vocab perception becomes the differentiator; 3DGraphLLM
quantifies the GT-to-predicted cliff at ~11 accuracy points.

### Decision / Follow-up

Verify GT-marker removal (and the "scored differently" rule)
against the cloned 2026 dev kit - top proposed next step.
