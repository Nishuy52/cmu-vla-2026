> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# NROS Team (1st place, 2025) - double champion, method
partially disclosed

One-liner: the 2025 CMU-VLA-Challenge winner (sim final 36.03,
overall 44.26, also won the real-robot track) is fielded by a
Chinese SLAM/robotics lab with no prior VLN/VLA track record. As of
2026-07-11, a second, official university news article (found via
a Wayback Machine snapshot after the live page went dead) adds real
substance: four team-member names, an English framework name
("Vision-Language Autonomous Agent Framework"), named techniques
(scene-graph construction, a multi-modal frontier-exploration
scoring mechanism), and an explicit mapping of their solved task
types onto the challenge's three official categories. Still no
paper, no code repo, and no VLM/LLM backbone model name.

## What it is

- Team from the **Networked RObotics and Systems Lab (NROS /
  nROS-Lab)**, **Harbin Institute of Technology, Shenzhen (HITSZ)**.
  Lab site: https://www.nrs-lab.com/
- Departmental affiliation has three variants across sources found
  so far; the most authoritative (an official HITSZ university news
  article, not the lab's own blog) says **智能学部智能科学与工程学院**
  ("Faculty of Intelligence, School of Intelligent Science and
  Engineering"), Shenzhen campus. Chen Haoyao's own faculty page
  (http://homepage.hit.edu.cn/chenhaoyao) lists him under "School of
  Intelligent Science and Engineering (Shenzhen)" too, which agrees
  with this variant. Older bios say "School of Mechanical
  Engineering and Automation" or "Department of Automation" - treat
  those as outdated or mistranslated; **School of Intelligent
  Science and Engineering is the current, best-sourced name.**
- Lab founded 2009, directed by **Chen Haoyao (陈浩耀)**, a full
  professor at HITSZ since 2020 (associate professor 2012-2019,
  assistant professor 2009-2012; PhD 2004-2009, Dept. of Precision
  Machinery and Precision Instruments, USTC).
  Source: https://www.researchgate.net/profile/Haoyao-Chen-2
- The lab's normal output is **not** vision-language navigation.
  Its GitHub org (https://github.com/HITSZ-NRSL) and publication
  list are SLAM, visual-inertial odometry, LiDAR/camera
  calibration, dense mapping, terrain-aware planning, and aerial
  manipulation - e.g. Dynamic-VINS (RA-L 2022), RIM (ICRA 2024),
  IGLOV (TMECH 2023), RCPCC (ICRA 2025), DynaLOAM, Det6D,
  Terrain-aware-planning. This CMU-VLA-Challenge win looks like a
  one-off entry built on top of that SLAM/perception base, not an
  established VLN research line.
- Per the lab's own announcement, the team "won the double
  championship in both the simulation-environment track and the
  physical-robot-platform track" ("斩获模拟环境和实物平台赛双冠").
  Source: https://www.nrs-lab.com/2025/10/13/热烈祝贺nros实验室代表队斩获cmu-vision-language-autonomy挑战赛（cmu-vla-challenge/

## What is known about the method

Two Chinese-language sources exist: the lab's own announcement post
(2025-10-13, previously the only source) and, found in this pass, a
higher-authority **official HITSZ university news article**
("捷报！陈浩耀教授团队在国际顶级科技赛事CMU-VLA大赛中夺冠",
published 2025-11-07). The live article page is now dead (404/410 -
apparently purged from the site's rolling news archive), but it was
recovered via a **Wayback Machine snapshot**
(http://web.archive.org/web/20251215081110/https://www.hitsz.edu.cn/article/view/id-380378.html,
captured 2025-12-15). Findings below are tagged by source.

### From the lab announcement (2025-10-13)

- The team describes its work as an **independently, self-designed
  "vision-language-action large-model algorithmic framework"**
  ("独立自主设计了视觉语言动作大模型算法框架").
- Three capability areas are named as the technical problems
  solved:
  1. natural-language-instruction scene understanding
     ("基于自然语言指令的场景理解"),
  2. object spatial-semantic recognition
     ("物体空间语义识别"),
  3. autonomous navigation-action generation and task execution,
     grounded in semantic and spatial relationships
     ("基于语义与空间关系的导航动作自主生成与任务执行").
- The post frames the win as overcoming "generalization and
  autonomy bottlenecks in real-world scenarios"
  ("现实场景中的泛化与自主能力瓶颈") - i.e. their pitch is sim-to-
  real transfer, consistent with winning both the sim and the
  real-robot tracks.

### From the official HITSZ university article (2025-11-07, via Wayback)

This is a materially richer source than the lab blog. Full recovered
body text (Chinese), for reference:

> 【哈工大（深圳）宣】（智能学部 文/图）近日，由IEEE机器人与自动化
> 协会主办的IEEE智能机器人与系统国际会议（IROS 2025）在杭州举行，
> 会议公布了由卡内基梅隆大学主办的国际大赛"CMU-VLA"竞赛结果，在
> 激烈的仿真对抗与真机部署环节中，哈工大深圳校区智能学部智能科学与
> 工程学院陈浩耀教授指导的"NROS"团队脱颖而出，以性能指标领先第二名
> 30%的优异成绩荣获冠军。针对竞赛中的目标计数、物品查找、视觉指令
> 导航三类任务，NROS团队（成员刘笑、李灵皓、曹宇豪、朱骥）自主构建
> 了一套"视觉语言自主智能体算法框架"（Vision-Language Autonomous
> Agent Framework）。该框架融合视觉、语言与空间感知信息，通过视觉
> 语言大模型赋能，使机器人能够理解自然语言指令并在复杂环境中自主
> 执行任务。团队创新性地引入场景图构建与多模态前沿探索评分机制，
> 实现了高效的目标识别与路径规划。系统在仿真与真实环境中均展现出
> 卓越的泛化与稳定性能，为视觉语言自主系统提供了新的解决方案。
> （编辑 谢梁晖 审核 张惠屏 陈南坤）

Extracted facts:

- **Team-member names (new - not found anywhere before this pass)**:
  刘笑 (Liu Xiao), 李灵皓 (Li Linghao), 曹宇豪 (Cao Yuhao), 朱骥
  (Zhu Ji). No individual profile pages, GitHub accounts, or arXiv
  listings under any of these names have been checked yet - see
  "What is unknown" below.
  - Note: 谢梁晖 / 张惠屏 / 陈南坤 (editor/reviewer credits at the
    end) are university news-office staff, **not** team members -
    do not conflate them with the four names above.
- **Performance margin, now source-confirmed** (not just inferred
  from the leaderboard numbers): "以性能指标领先第二名30%的优异成绩
  荣获冠军" - won with performance metrics **leading 2nd place by
  30%**. (For reference, 44.26 vs. ReasonX's 34.58 is a ~28% gap,
  consistent with this "~30%" framing - the two independently
  reported numbers corroborate each other.)
- **The three competition task types the team explicitly names are
  a direct, one-to-one match to our own three question types**:
  1. 目标计数 (object counting) -> **numerical** questions
  2. 物品查找 (object finding) -> **object-reference** navigation
  3. 视觉指令导航 (visual instruction navigation) ->
     **instruction-following** navigation
  This is more precise than the "three capability areas" language
  in the lab's own post - here the article maps method components
  directly onto challenge task categories, not just abstract skills.
- **Framework name with an English gloss, for the first time**:
  "视觉语言自主智能体算法框架" = **"Vision-Language Autonomous Agent
  Framework"**. (Differs slightly from the lab post's looser
  "vision-language-action large-model algorithmic framework" -
  treat "Vision-Language Autonomous Agent Framework" as the more
  deliberate/citable name, since it appears with an English
  translation the team itself likely supplied to the news office.)
- **Named techniques (new specifics, still short of a paper)**:
  - "融合视觉、语言与空间感知信息" - fuses vision, language, and
    spatial-perception information.
  - "通过视觉语言大模型赋能" - powered by a vision-language large
    model (VLM backbone confirmed to exist; **no specific model
    name given** - not GPT-4V, not Qwen-VL, not any named model).
  - "场景图构建" - **scene-graph construction** (first concrete
    representation named; consistent with the "detect -> scene
    graph -> reason" pattern flagged in our own `docs/prior_art/README.md`
    as the likely-strong baseline shape).
  - "多模态前沿探索评分机制" - a **multi-modal frontier-exploration
    scoring mechanism** - reads as frontier-based exploration
    (a standard technique in robot exploration/coverage planning)
    with a scoring function that incorporates multi-modal (vision +
    language) signals, used for "高效的目标识别与路径规划" (efficient
    object recognition and path planning).
  - "系统在仿真与真实环境中均展现出卓越的泛化与稳定性能" - the system
    showed strong generalization and stability in both sim and real
    environments - reiterates the sim-to-real pitch from the lab
    post, now from an independent source.
- The article includes an embedded image captioned/titled
  "荣誉证书" (honor certificate, filename `1.png` /
  `1762479493157051624.png`) plus several sidebar thumbnail images
  for unrelated "hot news" items on the same page (a museum-visit
  photo, a job fair photo, a UN University AI-literacy-camp student
  photo) - **those three sidebar photos are not about the NROS win**,
  confirmed by inspecting them directly; do not mistake them for
  team/demo photos in a future pass. The honor-certificate image
  itself was not captured by the Wayback crawler (checked via the
  Wayback `available` API for that exact image URL - no snapshot
  exists), so it could not be viewed in this pass.

### Still not found (re-confirmed this pass)

- The team was slated to present at the **IROS 2025 "AI Meets
  Autonomy" workshop on October 24, 2025**. That talk is still the
  single most likely source of a real architecture diagram, model
  names, or ablations. This pass fetched
  https://www.ai-meets-autonomy.com/iros-workshop-2025 directly: it
  confirms the Oct 24, 2025, 1:30-5:00pm, Room 210C slot and that
  "winning teams will present their methods through poster sessions
  and short spotlight talks" with an interactive demo - but the page
  itself has **no recordings, videos, or slide links**, and lists no
  team named NROS in its visible content (only speaker-homepage
  links for invited talks, e.g. wangwenshan.com, haochenz11.github.io,
  frc.ri.cmu.edu/~zhangji, etc. - none NROS-affiliated).
- No model names (LLM/VLM backbone), no detector name, no map
  representation beyond "scene graph," no robot platform model, and
  no architecture figure/diagram have been found in any source.

## What is unknown / how to find out

- **Method paper**: not found. Chen Haoyao's usual publication
  venues (IJRR, RA-L, T-RO, TMECH, ICRA per
  https://www.nrs-lab.com/publication/) are SLAM/mapping-focused;
  a search for a 2025-2026 VLN/VLA paper under his name returned
  nothing (arXiv, ResearchGate, OpenReview, Google-Scholar-adjacent
  search). Now that four team-member names are known (刘笑/Liu Xiao,
  李灵皓/Li Linghao, 曹宇豪/Cao Yuhao, 朱骥/Zhu Ji), the next search
  pass should try arXiv/Google Scholar/OpenReview under **each
  student's name** individually, not just Chen Haoyao's - a student-
  first-author challenge paper is common and would not surface under
  the advisor's name alone. None of the four names were searched
  individually in this pass (time-boxed) - do this next.
- **Code repo**: none found under https://github.com/HITSZ-NRSL
  (re-checked this pass: 44 repos as of 2026-07, still all SLAM/
  mapping/firmware/point-cloud - Dynamic-VINS, RIM, IGLOV, RCPCC,
  DynaLOAM, RefineNet, etc. - nothing VLN/VLA-shaped) or the
  director's personal account https://github.com/HitszChen. Also
  not checked yet: personal GitHub accounts under the four student
  names above (not attempted this pass - guessing GitHub handles
  from Chinese names is unreliable without a starting point such as
  an ORCID or lab roster page).
- **IROS workshop talk**: presentation was scheduled for
  2025-10-24 at the "AI Meets Autonomy" workshop. This pass fetched
  https://www.ai-meets-autonomy.com/iros-workshop-2025 directly
  (it resolves fine - the earlier 404 was specifically on the
  `/cmu-vla-challenge` route, not the workshop-year pages) and
  confirmed the schedule slot but found no recording, video, or
  slide link, and no team names. The challenge-results route itself
  (https://www.ai-meets-autonomy.com/cmu-vla-challenge) still 404's
  on both headless fetch and a direct browser-UA curl in this pass - this
  is very likely a genuinely dead/renamed route now, not just a
  fetch-tool limitation (confirmed by curl, which is not subject to
  the headless fetcher's JS-rendering restriction).
- **Team roster**: **four names now known** (see above), from the
  official HITSZ university news article, recovered via Wayback
  Machine after the live article page went dead. Their individual
  roles within the team (perception vs. planning vs. LLM-prompting,
  etc.) are not stated anywhere found.
- **VLM/LLM backbone model name**: still unknown. The official
  article says only "通过视觉语言大模型赋能" (empowered by a vision-
  language large model) with zero model name - not GPT-4V, not
  Qwen-VL, not LLaVA, nothing. This is the single highest-value
  missing fact for anyone trying to reproduce the approach.
- **Note - a red herring to rule out**: a different HITSZ group
  (School of Computer Science and Technology, "JiuTian-VL" /
  iLearn-Lab) published **CogVLA** (NeurIPS 2025, a cognition-
  aligned Vision-Language-Action model for robotic manipulation,
  https://arxiv.org/html/2508.21046v3). This is the same
  university but an unrelated lab/department (manipulation, not
  navigation; no connection to Chen Haoyao or NROS found) - do not
  conflate it with the NROS Challenge entry when searching for "HITSZ
  VLA" work.
- **Robot platform (real-robot track)**: not specified in any
  source found. The lab's general hardware roster includes ground
  robots, tracked robots, and aerial platforms
  (https://www.nrs-lab.com/), but nothing ties a specific platform
  to the CMU-VLA-Challenge real-robot round.

## Key numbers

(As supplied by the task; not independently re-derived from a
leaderboard page - the challenge site's results route
(https://www.ai-meets-autonomy.com/cmu-vla-challenge) still 404's
on fetch as of this pass - confirmed dead via both headless fetch and a
direct browser-UA curl, and no Wayback snapshot was checked for that
specific route yet. These numbers should still be spot-checked
against a Wayback Machine snapshot of that URL if one exists.)

- 2025 final leaderboard, NROS Team: **1st place**, overall score
  **44.26**, sim-preliminary score **36.03**.
- Also placed 1st in the real-robot track (per the lab's own
  "double championship" claim); no separate numeric score for the
  real-robot round was found.
- For context, other 2025 finalists (all four advanced to the
  real-world round): 2nd ReasonX 34.58, 3rd CopyPasta 30.98, 4th
  Urban Robotics Lab @ KAIST 22.80.
- **Independent corroboration found this pass**: the official HITSZ
  university article states NROS won "以性能指标领先第二名30%的优异
  成绩" (leading 2nd place by 30% on performance metrics). This is
  consistent with, and independently corroborates, the 44.26-vs-
  34.58 gap against ReasonX (a ~28% relative gap) supplied by the
  task - two different sources (task-supplied leaderboard row vs.
  HITSZ's own news office) now agree on roughly the same margin.

## Takeaways for our 2026 module

- The single strongest, most concrete claim is **sim-to-real
  transfer as the selling point** ("overcoming generalization and
  autonomy bottlenecks in real-world scenarios," reiterated
  independently by the official HITSZ article as "卓越的泛化与稳定
  性能") - worth taking seriously as a design pressure: whatever we
  build should not overfit to Unity-sim-only cues (e.g. perfect
  odometry, clean point clouds) if a real-robot round is a
  possibility for us too.
- **The task-type mapping is now explicit, not just analogous.**
  The official HITSZ article names the three competition tasks the
  team solved - 目标计数 (object counting), 物品查找 (object
  finding), 视觉指令导航 (visual instruction navigation) - which map
  one-to-one onto our own three question types (numerical, object-
  reference, instruction-following). Combined with their named
  technique of **scene-graph construction** plus a **multi-modal
  frontier-exploration scoring mechanism**, this is reasonably
  strong external validation for the "detect -> scene/semantic
  graph -> LLM/VLM reasons over it -> frontier-based exploration for
  navigation" pattern already flagged as the likely-strong baseline
  shape in `docs/prior_art/README.md`. Still consistent, not novel,
  evidence - but now with two concretely-named components (scene
  graph; frontier-exploration scoring) worth treating as candidate
  building blocks rather than pure speculation.
- Do not over-index on this team as a technical model to copy: we
  still have no architecture diagram, no VLM/LLM backbone name, and
  no code. What changed this pass is a team roster (4 names) and a
  named framework/technique vocabulary - useful as search anchors,
  not yet as a design template.
- Practical action: set a recurring check (e.g. monthly during
  Phase 0/1) on:
  - https://www.nrs-lab.com/publication/
  - https://github.com/HITSZ-NRSL
  - https://github.com/HitszChen
  - https://www.ai-meets-autonomy.com/ (workshop recordings page)
  - arXiv/Google Scholar for each of: 刘笑 (Liu Xiao), 李灵皓 (Li
    Linghao), 曹宇豪 (Cao Yuhao), 朱骥 (Zhu Ji) - not yet done
    individually as of 2026-07-11.

## Targeted deep-dive (2026-07-11): channels searched

Per the task's specific new angles, the following were searched this
pass. Recorded here so they are not redone unless new time has
passed or a specific new lead emerges.

- **Bilibili** (`search.bilibili.com` via WebSearch, queries: "CMU
  VLA 挑战赛 冠军", "site:bilibili.com CMU VLA Challenge HITSZ",
  "bilibili CMU VLA 挑战赛 冠军"): **no relevant results**. Only
  unrelated esports/gaming/CS-course videos surfaced. No NROS demo
  or talk video found on Bilibili.
- **WeChat public-account mirrors** on sohu/163/qq/zhihu (queries:
  "哈工大深圳 CMU VLA 冠军 微信公众号", "weixin.qq.com 哈工大深圳 CMU
  VLA 冠军"): **no mirror article found**. No WeChat-originated
  writeup about this specific win surfaced on any of those mirror
  domains.
- **HITSZ official news** (`site:hitsz.edu.cn` search plus direct
  fetch attempts): **found** - see the official article covered
  above, recovered via Wayback Machine after the live page (id-
  380378) returned 404/410. This was the single biggest new find of
  this pass.
- **Lab members / individual student search**: attempted via the
  lab's people/homepage and the four now-known names, but **did not
  reach the point of checking each name individually** against
  GitHub/arXiv in this pass (noted as next step above) - do not
  mark this channel as exhausted.
- **Zhihu/CSDN** ("CMU VLA Challenge 2025 方案 OR 冠军 方法", "陈浩耀
  nROS 视觉语言动作 IROS 论文 2025 2026"): **no method writeup
  found**. Only general IROS-2025-paper-roundup articles surfaced
  (e.g. a Zhihu "IROS 2025 精选论文盘点" post) and none mention NROS
  or Chen Haoyao.
- **IROS 2025 workshop talk recording** (YouTube + Bilibili, plus
  direct fetch of https://www.ai-meets-autonomy.com/iros-workshop-2025):
  **no recording found**. Only an unrelated IROS'24 YouTube talk by
  Anand Singh surfaced (wrong year, different context). The 2025
  workshop page itself confirms the schedule slot but has no
  embedded recording/slide links.
- **`ai-meets-autonomy.com/cmu-vla-challenge` route**: re-confirmed
  dead (404) via both headless fetch and a direct browser-UA curl - this
  looks like a genuinely retired/renamed URL, not a JS-rendering
  artifact. Not re-checked against Wayback Machine in this pass
  (unlike the HITSZ article, which was successfully recovered that
  way) - **worth trying Wayback for this specific URL next time.**
- **HITSZ-NRSL GitHub org**: re-checked (44 repos as of 2026-07 vs.
  32 previously noted) - still zero VLA/VLN-shaped repos.
- Incidentally, while searching, an **unrelated but adjacent finding**
  surfaced: the CMU-side 3rd-place team ("CopyPasta") was identified
  by name via an MRSD newsletter article
  (https://labs.ri.cmu.edu/mrsd-news/) as four MRSD Class-of-2026
  students - Sreeharsha Paruchuri, Ishita Gupta, Parth Singh, Daksh
  Adhar. This is **not** the NROS team and is out of scope for this
  dossier, noted here only so a future pass doesn't waste time
  re-discovering it if a CopyPasta dossier is ever wanted.

## Targeted deep-dive (2026-07-11, pass 2): per-member identity hunt

Per-member GitHub/arXiv/Scholar search for the four names, as the
next step flagged at the end of the previous pass. This pass found
**one confirmed individual identity** and independent roster
corroboration for two more names, but still no method paper or code.

### Confirmed: Liu Xiao (刘笑) = GitHub `liuxiao916`

- Found via `gh api search/users?q=liu+xiao+HITSZ` -> single hit
  `liuxiao916`, bio "Joint doctoral candidate at HITsz and CityU",
  location Shenzhen. Personal site (blog link on the GitHub
  profile, `liuxiao916.github.io`, redirects to custom domain
  `liuxiao916.me`) **confirms this is the same person named in the
  HITSZ article**:
  - PhD candidate, Harbin Institute of Technology Shenzhen, advised
    by **Professor Haoyao Chen** (NROS-Lab) - i.e. the lab director
    named in the university article - joint PhD with City University
    of Hong Kong under Professor Lu Liu.
  - Honors list on the site states, verbatim: **"Champion, 2nd CMU
    Vision-Language-Autonomy Challenge, IROS 2025"** - direct
    first-party confirmation of the win, independent of the HITSZ
    news article.
  - Publications listed (none are a VLA-challenge method paper):
    RGB-Thermal Visual Place Recognition via Vision Foundation Model
    (IROS 2025, with Minghao Ye, Yu Wang, Lu Liu, Haoyao Chen);
    Terrain-Adaptive Planning of a Mobile Robot With a Multiaxis
    Gimbal System for Stable SLAM (Trans. on Field Robotics, 2025);
    Contrastive Learning-Based Attribute Extraction Method for
    Enhanced Terrain Classification (ICRA 2024); Hierarchical
    Adaptive Voxel-guided Sampling for Real-time Applications in
    Large-scale Point Clouds (arXiv, 2023). ORCID
    (0009-0005-6578-7224, linked from the site) lists only the
    RGB-Thermal VPR paper via the ORCID public API - **confirms no
    VLA-challenge method paper has been indexed anywhere under her
    name as of this pass.**
  - Site's Projects and Blog-Posts pages are still template
    placeholders ("Todo...", "Future Blog Post") - no writeup of
    the challenge method there.
  - Her Zhihu profile link (`zhihu.com/people/wen-dao-zhu-yao-33`,
    also on the site) returned HTTP 403 on fetch - not checked this
    pass (Zhihu blocks non-browser fetches); worth a manual check
    later.
- **GitHub repo evidence - still no method code found**: her account
  has a fork `liuxiao916/CMU-VLA-Challenge` of the official dev-kit
  template (`HaochenZ11/CMU-VLA-Challenge`), last pushed
  2025-08-21. Checked the fork's commit log directly: **all commits
  are upstream commits by Haochen Zhang** (the challenge organizer)
  - the fork has **zero divergent commits of its own**, and
  `ai_module/src/` still only contains the stock `dummy_vlm`
  scaffold. This means her team's actual solution was never pushed
  to this public fork (kept in a private repo/branch, or submitted
  via a non-forked repo per the submission form).
  - Cross-checked all 24 public forks of `HaochenZ11/CMU-VLA-
    Challenge` via `gh api .../forks`: none of the fork owners'
    usernames resemble NROS/HITSZ team members, and a spot-check of
    one other fork with a recent-looking `pushed_at`
    (`YatsenRobot/CMU-VLA-Challenge`) showed the same pattern - all
    commits are upstream-only, no team code. **This channel (public
    GitHub forks of the template repo) is now exhausted** - no team
    appears to have pushed a real solution to a public fork.
  - Her other repos include forks of `ETPNav` (VLN-in-continuous-
    environments, TPAMI 2024) and `ESAM` (EmbodiedSAM, ICLR 2025
    Oral - online 3D instance segmentation) plus a personal
    `isaac_sim_pointcloud_tool` - consistent with, but not proof of,
    a scene-graph/3D-segmentation pipeline; these are read-only
    forks with no team-specific commits, so treat as a **plausible
    tooling-interest signal only**, not evidence of the actual
    method.

### Roster corroboration: Cao Yuhao and Zhu Ji confirmed as lab members (independent of the HITSZ article)

- The NROS lab's own people page
  (https://www.nrs-lab.com/people-2/) independently lists **曹宇豪
  (Cao Yuhao)** as a 2024-level Master's candidate and **朱骥 (Zhu
  Ji)** as a 2025-level Master's candidate - this corroborates the
  HITSZ article's roster from a second, independent source (the
  lab's own site, not the university news office). Neither has an
  individual GitHub/personal-page link listed on that page.
- **李灵皓 (Li Linghao) was NOT found** on this lab people-page
  listing (checked the PhD-candidate, Master's-candidate, and
  research-assistant sections) - this is a discrepancy worth
  flagging: either the roster page is incomplete/outdated, the name
  has a transliteration mismatch, or this member has since left/was
  visiting from elsewhere. Treat as **unconfirmed via this channel**
  pending a direct look at the page in a future pass (this pass only
  had it summarized via headless fetch, not read in raw HTML).
- GitHub user search (`gh api search/users`) for `cao yuhao`,
  `zhu ji`, and `li linghao` variants returned no accounts with a
  bio/location tying them to HITSZ or this lab - none of the
  candidate hits (`ZorAttC`, `CanvasCao`, `jizhu1023`, etc.) could
  be confirmed as the right person from public profile fields alone.

### Still not found this pass

- No arXiv, Google Scholar, or Semantic-Scholar-adjacent hit for a
  2025-2026 paper titled or themed around "Vision-Language
  Autonomous Agent Framework," or combining "scene graph" +
  "frontier exploration" + HITSZ/Chen-Haoyao/NROS authorship.
  (Semantic Scholar API was not queried this pass - not reached
  before the fetch budget ran low; still worth trying next time.)
- VLM/LLM backbone name: still unknown.
- Method paper and code repo: still not found - now with stronger
  confidence the code was never pushed to a public GitHub fork of
  the template (see above), so any future search should look for a
  **private-repo-turned-public** or a **separate, non-forked repo**
  rather than continuing to scan forks of the template.

## Sources

- **Xiao Liu's personal academic site (confirms her identity,
  advisor, and the challenge win first-party)**:
  https://liuxiao916.github.io/ (custom-domain mirror,
  same content: http://liuxiao916.me/ - note: HTTPS on the custom
  domain has a certificate mismatch, fetch it over plain HTTP or via
  the `.github.io` URL instead)
- Xiao Liu's ORCID (only one paper indexed, not the VLA-challenge
  method): https://orcid.org/0009-0005-6578-7224
- Xiao Liu's GitHub: https://github.com/liuxiao916 - fork of the
  challenge template with zero divergent commits:
  https://github.com/liuxiao916/CMU-VLA-Challenge
- NROS lab people page (independent roster corroboration for Cao
  Yuhao and Zhu Ji as Master's candidates; Li Linghao not found
  here): https://www.nrs-lab.com/people-2/
- Official challenge dev-kit template repo, used to cross-check all
  public forks for a divergent/team solution (none found):
  https://github.com/HaochenZ11/CMU-VLA-Challenge
- Lab announcement (2025-10-13):
  https://www.nrs-lab.com/2025/10/13/热烈祝贺nros实验室代表队斩获cmu-vision-language-autonomy挑战赛（cmu-vla-challenge/
- **Official HITSZ university news article (2025-11-07) - the
  strongest source found to date.** Live URL is now dead (404/410
  as of 2026-07-11):
  https://www.hitsz.edu.cn/article/view/id-380378.html
  Recovered via Wayback Machine snapshot (2025-12-15):
  http://web.archive.org/web/20251215081110/https://www.hitsz.edu.cn/article/view/id-380378.html
- Lab homepage (director, research areas, hardware roster,
  publication link): https://www.nrs-lab.com/
- Chen Haoyao's HITSZ faculty page (confirms current department
  name "School of Intelligent Science and Engineering (Shenzhen)";
  redirects from `faculty.hitsz.edu.cn/chenhaoyao`):
  http://homepage.hit.edu.cn/chenhaoyao
- Lab GitHub org (re-checked 2026-07-11, 44 repos, no VLA/VLN repo
  found): https://github.com/HITSZ-NRSL
- Director's personal GitHub (no VLA/VLN repo found):
  https://github.com/HitszChen
- Director's ResearchGate profile (career/education background):
  https://www.researchgate.net/profile/Haoyao-Chen-2
- IROS 2025 "AI Meets Autonomy" workshop page (schedule confirmed;
  no recordings/slides/team names found):
  https://www.ai-meets-autonomy.com/iros-workshop-2025
- Challenge workshop results route (still 404 on fetch as of this
  pass; recheck manually or via Wayback):
  https://www.ai-meets-autonomy.com/cmu-vla-challenge
  https://www.ai-meets-autonomy.com/
- Unrelated HITSZ VLA work, noted only to avoid conflation:
  CogVLA (NeurIPS 2025), https://arxiv.org/html/2508.21046v3
- Adjacent, out-of-scope finding (CMU 3rd-place team "CopyPasta"
  member names, not NROS): https://labs.ri.cmu.edu/mrsd-news/
- Cross-reference: this repo's own baseline analysis, which the
  NROS task-type mapping and technique split is consistent with:
  `docs/prior_art/README.md`
