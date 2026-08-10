# Submission plan (11 Aug 2026)

The deadline is 15 Aug 2026 AoE. Multiple submissions are allowed and
the highest score counts. Submit a measured baseline early, then
replace it only when a later tree measures better.

## The decision is an instruction-following decision

The rubric weights the categories 36 / 12 / 3 (`docs/challenge_brief.md`).
A change of 0.10 in the instruction-following mean is worth 3.6 points.
The same change in object_reference is worth 1.2, and every measured
object_reference difference between our trees is under 0.1 points.

**Choose the submission tree on the instruction-following mean.**

## Measured trees

| tree | IF mean | OR mean | IF points | note |
|---|---|---|---|---|
| 0d20d6c (pre-batch-6) | 0.4928 | not measured | 17.74 | The wedge defect is present. The vehicle freezes on 13 of 20 slots. |
| f52c908 (mid) | 0.3235 | 0.0186 | 11.65 | |
| 5786dd4 (wedge + throttle) | 0.3696 | not measured | 13.31 | Mechanisms correct. Five clean route completions. |
| f8002d4 (+#213) | not measured | 0.0265 | — | Best object_reference mean recorded. |

Sample sizes: instruction-following n=23 clean matched questions;
object_reference n=26 across two groups.

## The open question

The old tree measures 4.44 points higher on instruction-following, but
its vehicle freezes on most slots. Two of seventeen questions in a
frozen generation scored 1.000 because the stopped vehicle sat inside
the rubric region. That is park luck. It does not transfer to unseen
scenes, and its size is not measured.

The newest tree drives correctly but scores lower. Its deficit is
attributed (#202): 35 percent to the resolver (#215) and 53 percent to
the perception track flicker (#217). Both fixes exist and are verified
or in verification. Neither has been measured live.

## The plan

1. Land #215 and #217. Gate, verify, push.
2. Run one instruction-following sweep on that tree, with the same
   matrices every generation has used (`if8_20260808_g1` to `g3`).
   The comparison set is the table above. About 7.5 hours.
3. Submit the tree with the highest measured instruction-following
   mean. If the fixed tree beats 0.4928, submit it. If it does not,
   submit 0d20d6c and record why.
4. Keep the image build ready so a submission takes minutes, not hours
   (`docs/submission_docker_guide.md`).

## Baseline submission

Do not wait for step 3 to submit for the first time. Build and submit
the best measured tree now. A submitted score cannot be lost, and a
later submission replaces it only if it scores higher.

## Risks

- The image build needs about 28 GB and the host disk sits at 92
  percent. Free space before the build, or prune the old image after
  tagging.
- The build downloads about 7 GB on a cold cache. The layer cache from
  the 19 Jul build makes a same-base rebuild much faster.
- A scored dry-run needs the simulator and a GPU. Reserve time for it.
