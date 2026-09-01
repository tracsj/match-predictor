---
name: wrap-up
description: End-of-session capture. Summarises the session, appends skill learnings, commits this session's work. Consolidation (promotion, pruning, CLAUDE.md accuracy, permissions audit) belongs to /housekeeping, not here.
---

Before starting, read [wrap-up learnings](learnings.md) and apply any guidance from the Staging section.

Before we close this session:

1. **Session summary** — 3–5 bullets on what we did today.

2. **Registry reconciliation** — *the project-specific step, and the one most likely to be skipped.*

   List every hypothesis, model configuration, feature set, threshold or market variant **evaluated this session**, including ones abandoned after a single look. For each, confirm it appears in `docs/PROGRAMME.md` and that the running count is incremented.

   The risk is not a dishonest entry. It is a configuration tried casually mid-session, found uninteresting, and never written down — which quietly understates the search and inflates whatever eventually survives it.

   **Most of this step's real work is arguing things DOWN, not up** — three sessions running, so expect it. Dry runs, grading controls, benchmark measurements, null arms and re-slices of a settled result all look exactly like evaluations in a log. **The question that settles every one of them is what the thing was *searching for*.** A control whose result nobody is hoping for cannot widen a search. A re-slice of an existing run by a new dimension is a new cut, not a new configuration. Prices measured without a model fitted are a fact about the market.

   **Never record the outcome as a bare "nothing else was evaluated"** — that is indistinguishable from having skipped the step. Enumerate what was considered and why each does not count, in a table in `docs/PROGRAMME.md`. The 2026-08-17 third session ran **eight** such things beside one real evaluation, and the table naming them is the record that the count of 48 is honest.

   Then check: does any hypothesis whose status is `running` now have a result? Move it to `settled` with a link, or to the graveyard.

3. **Learnings** — for each skill used this session, either append a dated entry under `## Staging` in that skill's `.claude/skills/<name>/learnings.md`, or explicitly confirm there is nothing to capture.

   **Insert after the `## Staging` heading — do not append to end of file.** A learnings file may carry a second section (`## Archive`, `## Incident archive`), and a blind append lands the entry where nobody will read it.

   **Append only — decide nothing.** Promotion, pruning and the 30-day sweep moved to `/housekeeping` on 2026-09-01. If an entry looks worth promoting, say so inside it and name where you think it belongs; the weekly pass rules on it with every repo in view.

   **Never restate a size, count or backlog claim you inherited.** Measure it and quote the number you got, or leave it out. Correct a wrong inherited claim in place rather than adding a fresh entry beside it.

4. **Test suite** — Run `uv run pytest -q`. The harness self-tests are what make every number in this repo trustworthy: the cheater probe, the poisoned-split leak guard, the closed-form margin check, the published-band check on the market. **A failure here is not flaky — it means something real broke.** Do not close a session on a red suite without saying so explicitly.

5. **Stale working-doc scan** — Run `find . -maxdepth 1 -name "*.md" | sort` and `ls docs/*.md`. Known-persistent (skip): `CLAUDE.md`, `README.md`, `PROGRAMME.md`, any `PREREG*.md` (the original `PREREGISTRATION.md` plus per-hypothesis ones like `PREREG_PHASE6_NULL.md` — all frozen records, never edited retroactively), `FORWARD_LEDGER.md` (machine-written by the forecast workflow — never edit it by hand), and any `*_RESULT.md`. For each other file, ask the user: delete, move to the right subfolder, or keep with an explicit note added to CLAUDE.md. Do not silently skip or auto-delete.

6. **Next-session brief** — *the step whose absence is invisible until the next session opens cold.*

    Update the **"Where we are — read this first"** section at the top of `docs/PROGRAMME.md`: what was finished, what is next in order, and any open thread a fresh session would otherwise rediscover.

    It goes there rather than in a plan file because plan files live in `~/.claude/plans/` under generated names — there are dozens, and no future session has a reason to guess which one is this project's. `docs/PROGRAMME.md` is pointed at from `CLAUDE.md`, which auto-loads, so it is actually reachable.

    Write it for someone with no memory of the session. "Continue where we left off" is not a brief; the next concrete action, with the file it touches, is.

7. **Commit wrap-up changes** — Stage and commit any changes made during wrap-up. Scope every commit with a `--` pathspec after the `-m` flags; never `git add .`. For a multi-line message, Write to the scratchpad and use `git commit -F` — inline multi-line `-m` breaks on apostrophes. Run `git status` in **both** the project repo and `~/.claude/`.


## What moved to `/housekeeping` — 2026-09-01

Wrap-up is **capture**; `/housekeeping` is **consolidation**. Nothing was dropped.

| Was a wrap-up step | Now |
|---|---|
| Permissions audit | `~/.claude/hooks/git-tenancy-guard.py` blocks whole-worktree staging and `permission-rule-guard.py` blocks a banned allow rule at the moment of writing; `/housekeeping permissions` audits the files weekly. |
| Promote / prune Staging | `/housekeeping skills` |
| Cross-session pattern check | `/housekeeping patterns` and `learnings-meta` — they see every repo at once, which is what a cross-session pattern requires |
| CLAUDE.md accuracy, routing filters, `context-budget.py` | `/housekeeping claude-md` |
| File-size flags | `/housekeeping skills` |
| Context-file description audit | `/housekeeping context` |
| "Periodically run `/housekeeping`" | Deleted — a LaunchAgent rings it Mondays 09:15 |

**Why:** wrap-up was costing a median 41 tool calls per run (worst 93), of which the permissions audit was 23.6% and learnings judgement 30%, against 5% for the commit. And a per-session prune cannot reach a quiet repo, which is where entries actually rot. Baseline: `~/.claude/consolidation-baseline.md`.

**Resist re-adding steps here.** Every wrap-up in the system grew at 73–143 words per edit against a healthy 32–56, in all eleven repos, and a one-time trim regrew past its own peak in twenty days. A new check almost always belongs in `/housekeeping`.

## Verification

The wrap-up is complete when all of the following hold, checked rather than assumed:

- `uv run pytest -q` is green, or the failure is stated explicitly in the closing message.
- `git status` is clean in **both** `match-predictor` and `~/.claude/`.
- Every configuration evaluated this session appears in `docs/PROGRAMME.md`, and its running count moved by the right amount.
- The "Where we are" section names a **concrete next action and the file it touches**, not a topic.
- No `## Staging` section exceeds a handful of entries.

Run `git status` in both repos and `git log --oneline -3` in the project repo as the very last action, then output this report with each step marked ✓ or ✗:

```
Wrap-up complete:
✓/✗ Step 1 — Session summary (N bullets)
✓/✗ Step 2 — Registry reconciliation (N configs recorded; count X → Y)
✓/✗ Step 3 — Learnings
✓/✗ Step 4 — Test suite
✓/✗ Step 5 — Stale working-doc scan (clean / deleted or relocated N)
✓/✗ Step 6 — Next-session brief
✓/✗ Step 7 — Commit wrap-up changes
```

If any step is ✗, state why before closing.

## Final step — Capture learnings

If a wrap-up step behaved unexpectedly — skipped, fired incorrectly, or a pattern recurred — append a dated entry under `## Staging` in `learnings.md`. Do not promote or prune it here — `/housekeeping` owns that.
