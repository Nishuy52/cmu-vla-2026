# ReasonX (2nd place, CMU VLA Challenge 2025) - targeted
deep-dive findings

Scope: team identity, external footprint, talk recording, paper.
Code-level analysis of the ReasonX_SGTeam repo is owned by a
separate dossier - not duplicated here.

Confirmed inputs handed to this task (not re-derived): team
"ReasonX" placed 2nd, final score 34.58 (sim prelim 31.42), behind
NROS (44.26), ahead of CopyPasta (30.98). Repo:
https://github.com/Yuxin916/ReasonX_SGTeam (branch `vlm_baseline`).

## Correction to starting premise

The task briefing assumed ReasonX was an NUS (National University
of Singapore) team. Evidence below shows it is actually a joint
**NTU (Nanyang Technological University, Singapore) + NUS** team:
the lead (Yuxin916 / Cai Yuxin) is an NTU PhD student, while at
least one other member (Jie Chen) is an NUS PhD student. Treat
"NUS team" framing as inaccurate; "NTU-led, NUS-collaborator" is
the supported claim.

## Member identities

### Yuxin916 -> Cai Yuxin (team lead, repo owner)

- GitHub: https://github.com/Yuxin916 (login `Yuxin916`, display
  name on GitHub profile is "Tsaisplus", account id 43313010,
  created 2018-09-16).
- Commit email on the repo: `caiyuxin001220@gmail.com` - matches
  "Cai Yuxin" (surname Cai, given name Yuxin).
- Personal site: https://yuxin916.github.io/ (repo
  `Yuxin916/Yuxin916.github.io`, a fork of an academic-portfolio
  template - confirmed via WebFetch of the live page, not just the
  repo name).
  - Full name: **Yuxin Cai**.
  - Position: PhD student, Automated Driving and Human-Machine
    System Lab (**AutoMan**), School of Mechanical and Aerospace
    Engineering, **Nanyang Technological University (NTU)**,
    Singapore.
  - Advisor: Prof. **Chen Lv** (primary); co-supervised by Dr.
    **Wei-Yun Yau**, Institute for Infocomm Research (I2R),
    A*STAR (as an AGS scholar).
  - Institute email on the site: `caiy0039@e.ntu.edu.sg`.
  - Site news item (quoted by WebFetch extraction): "Our team
    ReasonX won the 2nd Place in CMU Vision-Language Autonomy
    Challenge and presented our work at IROS 2025!" - this is the
    strongest direct confirmation found that Cai Yuxin's team is
    the same ReasonX.
  - Separate news item: "We won the Best Paper Award (First
    Prize) for our paper 'COVLM-RL' at IEEE ITSC 2025!" - this is
    a **different, unrelated project** (autonomous-driving VLM+RL,
    not VLN/the challenge); do not conflate with ReasonX.
- Google Scholar: https://scholar.google.com/citations?user=_fEQHXQAAAAJ&hl=en
  Confirms NTU AutoMan-adjacent publication list (see Papers
  section below).
- ResearchGate: https://www.researchgate.net/profile/Yuxin-Cai-13
  (found via search, not separately fetched; listed as PhD
  Student, NTU Robotics Research Centre).

### J1dan -> likely a NUS student; exact real name NOT confirmed

- GitHub: https://github.com/J1dan (login `J1dan`, display name on
  profile is "鸡蛋" - Chinese nickname meaning "egg", not a real
  name; account id 92639702, created 2021-10-16).
- Commit email on the repo: `e1010693@u.nus.edu` - `u.nus.edu` is
  NUS's student email domain, confirming NUS affiliation, but the
  student-ID-style local part (`e1010693`) does not itself reveal
  a name.
- No personal homepage, blog, or bio found for this account. Public
  repos are unrelated side projects (Autonomous-Navigation-Pipeline,
  Frozen-lake RL, Highway-Decision-Making, a diffusion_policy fork).
- **Not confirmed** which "Chen Jie / Jie Chen" (if any) this
  account belongs to. Kept separate from the "Jie Chen" identity
  found below because no direct link (email, name string, or
  cross-reference) ties `J1dan` to that person - only the shared
  `u.nus.edu` domain is suggestive, which is weak evidence.

### "Chen Jie" clue (from Meeting_Minutes.md) -> probably Jie Chen, NUS MARMoT Lab

- Found via a chain of Google Scholar co-authorship, not via the
  repo: Cai Yuxin's Scholar profile lists a 2026 paper "ImagiNav:
  Scalable Embodied Navigation via Generative Visual Prediction and
  Inverse Dynamics" with co-authors "J Chen, Y Wang, R Bai, Y Cao,
  J Li, YW Yun, G Sartoretti".
- G Sartoretti = Prof. **Guillaume Sartoretti**, PI of the
  **MARMoT Lab** (Multi-Agent Robotic Motion Lab), Department of
  Mechanical Engineering, **National University of Singapore**.
  Lab site: https://www.marmotlab.org/ (bio page:
  https://www.marmotlab.org/bio.html, people page:
  https://marmotlab.org/people.html).
- The MARMoT Lab people page (fetched) lists a PhD student named
  **Jie CHEN** among current PhD students (alongside Junkai Lu,
  Derek Tan, Jimmy Chiun, Peizhuo Li, Joshua Taylor, Tanishq Duhan,
  Shivam Sood, William Teo). No bio detail, GitHub link, or
  homepage was present on that page for Jie Chen specifically.
- This gives a plausible identity for "Chen Jie": **Jie Chen, PhD
  student, MARMoT Lab, NUS Mechanical Engineering, advisor
  Guillaume Sartoretti** - who has at least one co-authored paper
  with Cai Yuxin (ImagiNav, 2026, not confirmed as ReasonX-related
  work itself). Mark as **plausible but not certain**: the
  Meeting_Minutes.md name and this Scholar/lab-roster match were
  never directly cross-referenced (no email, no explicit "ReasonX"
  mention on Jie Chen's own page - he appears to have none indexed).

### "Haoruo" clue -> Haoruo Zhang, NTU AutoMan Lab

- AutoMan Lab people page (https://lvchen.wixsite.com/automan/people,
  fetched) lists **Haoruo Zhang** as a current PhD student (2025
  Jan intake), in the same lab as Cai Yuxin, same advisor Chen Lv.
- Personal homepage: https://ezhanghz.github.io/
  - Full name: **Zhang Haoruo (张皓若)**.
  - Affiliation: NTU, College of Mechanical and Aerospace
    Engineering; advisor Prof. Lv Chen, AutoMan group.
  - Research interests: Human-Machine Interaction, Embodied
    Intelligence.
  - Site does not itself mention CMU VLA Challenge or ReasonX
    (checked, negative result) - the AutoMan-lab-roster match plus
    matching given name is the basis for this identification, not
    a direct self-reported claim.
- A LinkedIn profile "haoruo zhang - Baltimore, Maryland, United
  States" also surfaced in search; this is almost certainly a
  different, unrelated person (wrong geography/context) - do not
  use.

## Talk recording - not found (negative result)

Searched and checked directly, nothing found:

- YouTube search: "ReasonX" CMU VLA, "IROS 2025 CMU VLA 2nd place",
  "AI Meets Autonomy IROS 2025 workshop CMU VLA challenge" - no
  ReasonX-specific video surfaced. Only IROS'24 challenge videos
  (Anand Singh presentation, "Find the Refrigerator in the
  Lounge") came up, both pre-dating the 2025 result.
- Workshop site https://www.ai-meets-autonomy.com/iros-workshop-2025
  fetched directly: confirms the workshop happened Fri 24 Oct 2025,
  1:30-5:00pm, Room 210C, Hangzhou, and lists ~12 *invited general
  speakers* (Wenshan Wang, Ji Zhang, Haochen Zhang, Deva Ramanan -
  all CMU RI; Ting Cao - MSR; Jiangmiao Pang - Shanghai AI Lab;
  Angel Chang - SFU; Roozbeh Mottaghi - Meta FAIR; Siyuan Huang -
  BIGAI; He Wang - Peking Univ; Joseph Lim - KAIST; Feishi Wang -
  Peking Univ). These are workshop keynote speakers, NOT the
  challenge team presenters - the page only has a "Tentative
  Program" image placeholder for the actual challenge-results
  segment, no named team slots, no video embeds.
- `https://www.ai-meets-autonomy.com/cmu-vla-challenge` returns
  HTTP 404 as of this check (2026-07-11) - page may have been
  removed/renamed since the challenge concluded.
- No ReasonX/Cai-Yuxin/Jie-Chen video found on the CMU RI YouTube
  channel or elsewhere via search.
- Conclusion: **no publicly posted recording of ReasonX's IROS 2025
  talk has been found**. If CMU or the organizers post one later,
  it is not yet indexed by search as of 2026-07-11.
- For contrast/context only (not ReasonX): a CMU MRSD team
  "CopyPasta" (Sreeharsha Paruchuri, Ishita Gupta, Parth Singh,
  Daksh Adhar) placed 3rd and is documented in a CMU RI MRSD
  newsletter article (https://labs.ri.cmu.edu/mrsd-news/articles/)
  as having presented at the same workshop - that article does not
  name the 1st/2nd place teams either, and no video link was found
  there.

## Paper / preprint - no dedicated "ReasonX" paper found (negative result)

- Direct arXiv search for "ReasonX" + VLN/navigation: no matching
  title in 2025-2026 results. Nothing under that exact name exists
  on arXiv as of this check.
- Cai Yuxin's Google Scholar publication list (fetched in full) has
  no paper titled "ReasonX" and no paper that self-identifies as
  the CMU VLA Challenge submission. Full list for reference (title,
  year, co-authors), in case any is later confirmed as the
  ReasonX method paper:
  1. Context-Aware Driver Attention Estimation Using Multi-Hierarchy
     Saliency Fusion With Gaze Tracking (2024) - Z Hu, Q Li, K Su, C Lv
  2. **CL-CoTNav: Closed-Loop Hierarchical Chain-of-Thought for
     Zero-Shot Object-Goal Navigation with Vision-Language Models**
     (arXiv:2504.09000, Apr 2025) - X He, M Wang, H Guo, WY Yau, C Lv.
     Flagged as the **most plausible methodological precursor/
     sibling** to ReasonX: it is object-goal navigation (matches the
     challenge's object-reference task type), uses VLM + hierarchical
     chain-of-thought reasoning with closed-loop confidence feedback,
     tested in AI Habitat, and predates/overlaps the mid-2025 challenge
     window. **Not confirmed** as "the ReasonX paper" - no explicit
     cross-reference found linking this arXiv ID to the challenge or
     to the ReasonX_SGTeam repo.
  3. Transformer-based Multi-Agent RL for Generalization of
     Heterogeneous Multi-Robot Cooperation (2024) - X He, H Guo,
     WY Yau, C Lv
  4. TranSimHub: A Unified Air-Ground Simulation Platform (2025) -
     M Wang, Y Chen, A Pang, Y Xie, Z Ma, C Xu, K Jiang, D Wang, et al.
  5. Interaction-Aware Hierarchical Representation of Multi-Vehicle
     RL for Cooperative Control in Dense Mixed Traffic (2024)
  6. ImagiNav: Scalable Embodied Navigation via Generative Visual
     Prediction and Inverse Dynamics (2026) - J Chen, Y Wang, R Bai,
     Y Cao, J Li, YW Yun, G Sartoretti (the Jie-Chen/Sartoretti link,
     see above)
  7. VLMLight: Traffic Signal Control via VLM Meta-Control (2025)
  8. SysNav: Multi-Level Systematic Cooperation for Real-World,
     Cross-Embodiment Object Navigation (2026) - H Zhu, Z Li, Z Liu,
     K Guo, Z Lin, G Chen, C Lv, W Wang, J Oh, et al.
  9. OmniTraffic (2026)
  10. Robots as Tokens: Unified Diffusion Transformer for Multi-Robot
      Trajectory Generation (2026) - R Bai, J Chen, J Li, WY Yau, L Xie
  11. IntentNav: Learning Spatial-Visual Object Navigation from Human
      Demonstrations (2026) - Z Li, M Wang, M Bao, H Zhu, R Bai,
      D Zhao, Z Li, W Wang, WY Yau, et al.
  12. Goal2Pixel: Grounding Goals to Pixels for VLN (2026) - M Bao,
      H Xu, Z Li, J He, J Tang, C Lv, J Zhang, Y Xie, W Wang
  13. Dual-Interaction-Aware Cooperative Control for Mixed Traffic
      Congestion (2026)
  14. **COVLM-RL: Critical Object-Oriented Reasoning for Autonomous
      Driving Using VLM-Guided RL** (arXiv:2512.09349) - authors
      confirmed via direct PDF fetch: Lin Li, Yuxin Cai, Jianwu Fang,
      Jianru Xue, Chen Lv. This is the ITSC 2025 Best Paper (First
      Prize) mentioned on Cai Yuxin's homepage - confirmed unrelated
      to VLN/the challenge (autonomous-driving domain).
- None of items 2, 6, 8, 11, 12 (the navigation-adjacent papers)
  contain the string "ReasonX" or "CMU VLA" per the fetched
  abstracts/summaries. They read as a broader NTU-AutoMan /
  NUS-MARMoT navigation-research program that ReasonX likely grew
  out of or alongside, not as ReasonX write-ups themselves.
- Not checked directly (time-boxed out): full-text search inside
  each of these ~14 papers for a literal "ReasonX" or "CMU VLA
  Challenge" mention - only titles/abstracts were reviewed. If a
  future pass wants to fully rule this out, that's the next step.

## Other repos checked (via `gh api users/<login>/repos`)

- Yuxin916's 30 public repos are almost all forks tied to prior
  multi-agent RL / MARL work (HARL, epymarl, pymarl-oxwhirl,
  MARBLER, TransSimHub, etc.) plus recent items: `g1_speak`
  (own, Python, updated 2026-06-30), `Navigation-Physical-Experiment`
  (own, C++, "ROS 2 Jazzy autonomy stack ... bridge between Habitat
  and the real robot" - updated 2026-07-09, clearly active current
  work), `Qwen3-VL-finetune` (fork, updated 2026-01-29),
  `InternVL_annotation` (fork, updated 2025-10-09),
  `habitat-lab`/`habitat-sim` forks (updated mid-2025, annotation
  and multi-robot extensions) - consistent with an active VLN/
  embodied-AI research program, but none of these repos is named
  ReasonX or references the challenge in its description.
- `ReasonX_SGTeam` itself is a **fork** of
  `HaochenZ11/CMU-VLA-Challenge` (confirmed via
  `gh api repos/Yuxin916/ReasonX_SGTeam --jq '{fork, parent, source}'`).
  Its GitHub-reported "contributors" (HaochenZ11 / Haochen Zhang,
  nzantout / Nader Zantout) are very likely **upstream dev-kit
  authors carried over from the fork's shared history**, not
  ReasonX team members - do not conflate them with the team. Actual
  `vlm_baseline`-branch commit authors are only: `Yuxin916
  <caiyuxin001220@gmail.com>`, `J1dan <e1010693@u.nus.edu>`, plus
  the inherited upstream authors (Haochen Zhang, Nader Zantout).
- J1dan's 5 public repos are unrelated side/coursework projects
  (see above); no additional VLN work found there.

## Negative-result checklist (do not re-run these)

- WebSearch: `"ReasonX" CMU VLA Challenge 2025 IROS second place`
  - no team-specific hit.
- WebSearch: `site:linkedin.com "ReasonX" CMU VLA 2025` - no hit.
- WebSearch: `"e1010693@u.nus.edu" OR "J1dan" NUS github vision
  language navigation` - no hit tying the handle to a real name.
- WebSearch: `youtube "AI Meets Autonomy" IROS 2025 workshop CMU
  VLA challenge` - no ReasonX video.
- WebSearch: `arxiv "ReasonX" vision language navigation 2025 2026`
  - no matching paper title.
- WebSearch: `NTU news "Yuxin Cai" OR "Chen Lv" CMU VLA challenge
  2025 award` - no NTU press/news-office article found (only the
  personal homepage and Scholar hits already covered above).
- WebFetch: `https://www.ai-meets-autonomy.com/cmu-vla-challenge`
  - HTTP 404 as of 2026-07-11.
- WebFetch: `https://marmotlab.org/bio.html` - PI bio only, no
  member roster (roster is on the separate `/people.html` page,
  which was fetched successfully).

## Sources (all URLs cited above, deduplicated)

- https://github.com/Yuxin916
- https://github.com/J1dan
- https://github.com/Yuxin916/ReasonX_SGTeam
- https://github.com/HaochenZ11/CMU-VLA-Challenge
- https://yuxin916.github.io/
- https://ezhanghz.github.io/
- https://scholar.google.com/citations?user=_fEQHXQAAAAJ&hl=en
- https://www.researchgate.net/profile/Yuxin-Cai-13
- https://lvchen.wixsite.com/automan
- https://lvchen.wixsite.com/automan/people
- https://www.marmotlab.org/
- https://www.marmotlab.org/bio.html
- https://marmotlab.org/people.html
- https://www.ai-meets-autonomy.com/
- https://www.ai-meets-autonomy.com/iros-workshop-2025
- https://www.ai-meets-autonomy.com/iros-workshop-2024
- https://www.ai-meets-autonomy.com/cmu-vla-challenge (404)
- https://labs.ri.cmu.edu/mrsd-news/articles/
- https://arxiv.org/abs/2504.09000 (CL-CoTNav)
- https://arxiv.org/pdf/2512.09349 (COVLM-RL)
