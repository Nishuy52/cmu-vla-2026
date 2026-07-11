# Research friction log - 2025 results hunt (2026-07-11)

Every roadblock hit while identifying the CMU VLA Challenge 2025
winners and their methods, with the workaround that worked and
the rule to apply next time. Written to improve similar research
workflows; the generalizable rules are graduated to the repo
skill (see "Graduated rules" at the end).

## 1. Websites and the leaderboard

- **Challenge page taken down.** The live page
  (ai-meets-autonomy.com/cmu-vla-challenge) 404s - it was a
  Google Sites property and was removed/replaced when the 2026
  challenge launched. Both desktop-Chrome and Googlebot user
  agents got the same empty shell. Lesson: for a past-year
  competition page, go to the Wayback Machine FIRST; the live
  site is the wrong place to look for last year's results.
- **Content embedded as images.** The Wayback snapshots render
  the page text fine, but the leaderboard and workshop program
  are IMAGES hosted on lh3.googleusercontent.com with
  session-gated hotlink protection: HTTP 400/403 for every
  header combination (UA, Referer, Accept, sec-fetch-*), a
  third-party image proxy (images.weserv.nl) also failed, and
  the Wayback CDX index had no capture of the images
  themselves. ~1 hour of agent time went into this dead end.
  RESOLUTION: the image renders fine for a logged-out human
  browser; a human screenshot solved it in seconds.
  Rule: on Google Sites, expect content-as-images; the moment
  an image-locked artifact is identified, hand the URL to the
  human instead of iterating on fetch tricks.
- **JS SPAs generally.** Plain WebFetch cannot render them, but
  Wayback snapshots preserve the final DOM text - fetching the
  snapshot beats fetching the origin. Sitemap.xml/robots.txt
  probing on Google Sites yields nothing (also 404 shells).

## 2. YouTube / yt-dlp

- **Format-blocked video - SOLVED on the third pass.** The
  Anand Singh 2024 talk (youtu.be/-bIMTsnbuoY) exposes no
  downloadable formats or captions on default yt-dlp clients
  ("Only images are available for download" - SABR /
  format-signature / PO-token blocks across web, ios, mweb,
  tv), even after upgrading yt-dlp to 2026.07.04. FIX:
  `--extractor-args "youtube:player_client=android"` exposed a
  real playable format (itag 18) that downloaded fine; frames
  were then read visually (slides carried the method + exact
  scores). Rule: before declaring a YouTube video
  machine-inaccessible, cycle player_client (android, then
  others) - the block set differs per client. Captions may
  still be absent; a Whisper key (GROQ/OPENAI) would unlock
  audio transcription and is worth configuring.
- **Channel listing broken.** yt-dlp cannot enumerate a
  channel's /videos tab ("Unsupported lockup view model"
  extractor error, a known YouTube-layout incompatibility).
  Workaround: ytsearch queries with --flat-playlist instead of
  channel enumeration.
- **Win to remember.** The single richest method source of the
  whole task was a manually-captioned talk uploaded to a team
  member's PERSONAL channel (CopyPasta / Ishita Gupta,
  kAPltAaRPk4). Rule: once team member names are known, hunt
  their personal channels/pages before anything official.

## 3. Academic APIs

- **arXiv full-text API** intermittently 503s/times out;
  retries helped only sometimes. Fallback: WebSearch with
  site:arxiv.org queries.
- **Semantic Scholar search endpoint** returned 429 on every
  unauthenticated attempt. Workaround that WORKED: traverse the
  citation graph of a known anchor paper instead (papers citing
  VLA-3D, arXiv 2411.03540) - a challenge entry would almost
  surely cite the organizers' dataset. Better precision than
  keyword search anyway.

## 4. GitHub

- **Winners do not necessarily fork.** The 3rd-place repo
  (parths5/CMU-VLA-Challenge) is a COPY (parent: null), not a
  GitHub fork - fork-network enumeration alone misses real
  entries. Rule: also `gh search repos`/`gh search code` for
  distinctive strings (ROS topic names like
  way_point_with_heading / selected_object_marker, challenge
  name variants).
- **Fork compare pitfalls.** compare API fails when the fork's
  default branch is named differently (a fork used `noetic`);
  cross-repo compare 404s for non-fork copies; empty repos
  return HTTP 409. Check default_branch before comparing.
- **raw.githubusercontent.com** 404s if you assume branch
  `main` (url-kaist uses `master`). Fetch the repo metadata
  first.
- **Squashed exports hide history.** url-kaist's 19 commits all
  land on one day ("initial commit from old repo") - commit
  dates are not development dates; don't infer timelines from
  them.

## 5. Search engines and language

- **The decisive 1st-place fact existed only in Chinese.** No
  English source named the winner; the hit came from a query
  using native award vocabulary (冠军). Rule: search the native
  languages of plausible participant countries (zh, ko, ja)
  with native award terms (冠军/亚军/第二名, 우승/2위), not
  translated English phrases.
- **Acronym collision.** LinkedIn/web searches for "VLA"
  surface volleyball leagues; always qualify with CMU / IROS /
  robotics.
- **University-domain site: queries under-deliver.** Lab pages
  (nrs-lab.com) are often NOT under the university domain
  (hitsz.edu.cn); search the lab name directly rather than
  site:-scoping to the institution.

## 6. Harness / process

- **Images cannot reach subagents.** The leaderboard screenshot
  was twice sent to a background agent; only the text
  placeholder arrived. Images must be pasted into the MAIN
  conversation. (This cost one full round trip.)
- **Newsletters name teams, never methods.** Institutional
  blurbs gave names/placings only; method detail lived in
  personal artifacts (talk video, personal repos, member
  homepages).
- **Parallel 4-angle fan-out worked.** Forks / site+video /
  arXiv / news ran concurrently; the decisive facts came from
  two different angles, and negative results were
  cross-confirmed instead of trusted from one pass.
- **The verification pass earned its cost.** The
  most-mature-repo heuristic (url-kaist: best code, best lab
  pedigree, most stars) suggested a top-2 finish; a dedicated
  skeptic agent found zero corroboration, and the leaderboard
  later showed 4th place with no real-world points added.
  Rule: code quality, pedigree, and stars predict placing
  poorly; keep hypothesis and fact rigidly separated.
- **Subagents need placement context.** Early dossier agents
  ran before the leaderboard was known and could not do
  targeted winner searches; a second, winner-aware pass was
  needed. Rule: when a key fact lands mid-fan-out, consider
  messaging in-flight agents (SendMessage) or queueing a
  targeted second pass rather than accepting the blind-pass
  output.

## Graduated rules

Graduated into the pre-merge research workspace's agent-skill
nuances (its `.claude/skills/repo/SKILL.md`):

- Google Sites / image-locked artifacts: escalate to a human
  screenshot fast.
- Search native languages with native award vocabulary for
  team/award announcements.
- GitHub entry discovery: search by distinctive topic strings,
  not just the fork network.
