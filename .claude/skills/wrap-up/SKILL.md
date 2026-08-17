---
name: wrap-up
description: End-of-session capture for match-predictor. Summarises the session, records every hypothesis or configuration evaluated in the registry, captures learnings, applies promotion and pruning, and confirms CLAUDE.md accuracy. Use before closing any session. Do NOT use mid-session to record a single finding — write that straight into the relevant docs/hypotheses entry; and do NOT use it to run or settle a hypothesis, which belongs in its own pre-registered run.
---

Before starting, read [wrap-up learnings](learnings.md) and apply any guidance from the Staging section.

Before we close this session:

1. **Session summary** — 3–5 bullets on what we did today.

2. **Permissions audit** — Read `.claude/settings.json`. Flag any allow rules matching banned patterns: `node:*`, `python3:*`, `curl:*` (unscoped), `rm:*`, `uv:*` (the bare wildcard includes `uv run`, which is code execution), or any `*:*` wildcard. Remove or tighten before proceeding. Takes 10 seconds; prevents security regressions persisting across sessions.

3. **Registry reconciliation** — *the project-specific step, and the one most likely to be skipped.*

   List every hypothesis, model configuration, feature set, threshold or market variant **evaluated this session**, including ones abandoned after a single look. For each, confirm it appears in `docs/PROGRAMME.md` and that the running count is incremented.

   The risk is not a dishonest entry. It is a configuration tried casually mid-session, found uninteresting, and never written down — which quietly understates the search and inflates whatever eventually survives it. If nothing was evaluated, say so explicitly.

   Then check: does any hypothesis whose status is `running` now have a result? Move it to `settled` with a link, or to the graveyard.

4. **Learnings** — For each skill used, check if anything worked particularly well or went wrong. Either write a dated entry under `## Staging` in `.claude/skills/<name>/learnings.md` OR explicitly confirm nothing to capture. Then apply promotion + pruning:
   - Likely to happen again → promote into skill file or CLAUDE.md; delete the Staging entry
   - Repeated 3+ sessions → same
   - One-off → leave in Staging only
   - Older than 30 days, unpromoted → delete

   Staging stays lean — a handful of active observations at most.

5. **Test suite** — Run `uv run pytest -q`. The harness self-tests are what make every number in this repo trustworthy: the cheater probe, the poisoned-split leak guard, the closed-form margin check, the published-band check on the market. **A failure here is not flaky — it means something real broke.** Do not close a session on a red suite without saying so explicitly.

6. **Files** — Flag any file over 150 lines and suggest how to split it.

7. **Stale working-doc scan** — Run `find . -maxdepth 1 -name "*.md" | sort` and `ls docs/*.md`. Known-persistent (skip): `CLAUDE.md`, `README.md`, `PROGRAMME.md`, `PREREGISTRATION.md`, and any `*_RESULT.md`. For each other file, ask the user: delete, move to the right subfolder, or keep with an explicit note added to CLAUDE.md. Do not silently skip or auto-delete.

8. **Context file descriptions audit** — for any file added to or updated this session, check that its row in CLAUDE.md's "Where the detail lives" table names what is now inside it. If you find one stale or missing, check siblings — staleness clusters.

9. **CLAUDE.md accuracy** — Four passes:
   - *Additions*: anything new this session that should be documented? New data boundaries and new failure modes are the two that matter most here.
   - *Staleness*: is everything already there still accurate? Measured facts drift — a vendor changes coverage, a feed stops.
   - *Global*: did anything reveal a universal principle? Before adding to `~/.claude/CLAUDE.md`, pass this 3-question filter — all must pass: (1) behavioral guardrail that prevents a recurring mistake, not reference knowledge; (2) would cause a mistake in *typical* sessions; (3) plausibly applies in 2+ active repos. If not all three → route to skill gotchas, this CLAUDE.md, or `~/.claude/reference/troubleshooting.md`.
   - *Context budget*: run `python3 ~/.claude/bin/context-budget.py` before adding to `~/.claude/CLAUDE.md` **or** this repo's `CLAUDE.md`. Non-zero exit means a compaction pass is owed *before* the addition — keep the action in the hub, move mechanism and narrative to a spoke.

10. **Memory health** — Periodically (every 5–10 sessions) run `/housekeeping`. Skip if last run was recent.

11. **Commit wrap-up changes** — Stage and commit any changes made during wrap-up. Scope every commit with a `--` pathspec after the `-m` flags; never `git add .`. For a multi-line message, Write to the scratchpad and use `git commit -F` — inline multi-line `-m` breaks on apostrophes. Run `git status` in **both** the project repo and `~/.claude/`.

## Verification

The wrap-up is complete when all of the following hold, checked rather than assumed:

- `uv run pytest -q` is green, or the failure is stated explicitly in the closing message.
- `git status` is clean in **both** `match-predictor` and `~/.claude/`.
- Every configuration evaluated this session appears in `docs/PROGRAMME.md`, and its running count moved by the right amount.
- No `## Staging` section exceeds a handful of entries.

Run `git status` in both repos and `git log --oneline -3` in the project repo as the very last action, then output this report with each step marked ✓ or ✗:

```
Wrap-up complete:
✓/✗ Step 1  — Session summary (N bullets)
✓/✗ Step 2  — Permissions audit (clean / fixed N rules)
✓/✗ Step 3  — Registry reconciliation (N configs recorded; count X → Y)
✓/✗ Step 4  — Learnings (skills: X, Y — entries written / nothing to capture)
✓/✗ Step 5  — Test suite (N passed / N failed)
✓/✗ Step 6  — Large files (clean / flagged X)
✓/✗ Step 7  — Stale working-doc scan (clean / deleted or relocated N)
✓/✗ Step 8  — Context file descriptions audit (clean / updated N)
✓/✗ Step 9  — CLAUDE.md (N additions / nothing stale / global check)
✓/✗ Step 10 — Memory health (skipped / ran /housekeeping)
✓/✗ Step 11 — Changes committed; git status clean (project + ~/.claude)
```

If any step is ✗, state why before closing.

## Final step — Capture learnings

If a wrap-up step behaved unexpectedly — skipped, fired incorrectly, or a pattern recurred — append a dated entry under `## Staging` in `learnings.md`. Apply promotion + pruning per the global CLAUDE.md rules.
