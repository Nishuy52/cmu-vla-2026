# Claude Subscription Strategy

*Written 10 Jul 2026. Policy facts as reported early Jul 2026 — verify current terms in your Claude settings/console before spending money.*

## The Fable 5 clock

- **Fable 5 is included in paid plans only until 12 Jul 2026, 11:59 PM PT** (≈ 13 Jul 3 PM SGT). ~2 days left.
- Until then: up to **50% of your weekly usage limit** can go to Fable at no extra cost.
- After 12 Jul: Fable runs only on prepaid usage credits at API rates (**$10/M input, $50/M output**) — expensive; treat post-deadline Fable as a scalpel, not a daily driver.

## How to burn the Fable quota well (next 48h)

Spend Fable on work where frontier judgment compounds and cheaper models would produce rework:

1. **Architecture** — harden `docs/architecture.md`: exploration policy, perception fusion, scene-graph schema, LLM reasoning design, the 10-min time-budget strategy.
2. **Upstream repo deep-dive** — distill the autonomy stack + dummy ai_module into `docs/upstream_notes.md` so future (cheaper) sessions never re-read the raw code.
3. **Question analysis** — taxonomy over all 15 scenes' JSON; per-type answering strategy.
4. **Prior art** — SORT3D / 2025 winners / OpenEQA distilled into `docs/prior_art.md`.
5. **Core scaffolding** — the OS-independent `ai_module` core skeleton + interfaces + tests, so later Sonnet/Opus sessions fill in well-specified boxes.

Anti-pattern: don't burn Fable on bulk code generation, file plumbing, or web-scraping-style research — delegate those to subagents (Haiku/Sonnet) even during the window; Fable orchestrates.

## Max plan: recommendation

**Yes — upgrade to Max 5x ($100/mo) now, for two months, if you're currently on Pro.** Reasons:

- Upgrading **before 12 Jul** multiplies the weekly limit that the 50%-Fable allowance is computed from → materially more Fable in the closing window.
- The Jul 12 – Aug 15 stretch is a genuine crunch: sim iteration, code volume, multi-agent delegation. Pro limits will throttle you exactly when the deadline pressure peaks.
- $200 total for a shot at an IROS presentation + cash prize is asymmetric.

**Max 20x ($200/mo): not yet.** Upgrade mid-way only if you're hitting Max 5x weekly limits during Phase 2 sim iteration. **Downgrade after 15 Aug.**

## Post-Fable model mix (from 13 Jul)

| Work | Model | How |
|---|---|---|
| Orchestration, design decisions, plan review | Opus 4.8 | main session |
| Feature implementation | Sonnet 5 | `executor` subagent |
| Search/recon, doc lookups | Haiku 4.5 | `scout` subagent |
| Bulk/mechanical edits, test runs | Sonnet/Haiku | `mech-executor` |
| Verification of finished work | Sonnet 5 | `verifier` |
| Rare, genuinely hard design knots | Fable via usage credits | small, surgical prompts only |

Also free leverage: **the challenge allows online LLM APIs at test time** — the robot's reasoning module can call the Claude API. That's an API-billing question (separate from subscription), budget it when the architecture firms up; Haiku 4.5 for high-frequency perception queries, Sonnet 5 for the per-question reasoning call is the likely sweet spot.

## Weekly cadence (subscription hygiene)

- Weekly limits reset on a rolling basis — schedule heavy sim-iteration days right after a reset.
- Keep sessions focused; long meandering contexts eat quota. One task per session, handoff via `LOG.md` + docs.
- Sources: [usage-credit switch guide](https://www.digitalapplied.com/blog/claude-fable-5-usage-credits-july-7-pricing-guide-2026), [PCWorld on the restriction backlash](https://www.pcworld.com/article/3181897/claude-subscribers-are-furious-over-fables-new-restrictions.html), [webvise on post-Jul-12 credits](https://www.webvise.io/blog/fable-5-leaves-subscriptions-usage-credits).
