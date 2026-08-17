# wrap-up skill — learnings

## Staging

### 2026-08-17 (3rd session) — step 9's staleness pass caught two counts and one wrong conclusion
The staleness sub-step usually finds nothing. This session it found `CLAUDE.md` claiming **299 tests** against an actual 302, and "the four self-tests" against five, neither of which any other step would have surfaced — the suite was green both times, so nothing failed to draw attention to the drift.

More importantly it forced the question of whether a *conclusion* in `CLAUDE.md` was still true, not just a number. It was not: the gloss on the founding study's CLV depended on a null the session had just disproved. **The pass is worth more when read as "is every claim here still true?" rather than "have any numbers moved?"** — a number that drifts is cheap to fix, and a conclusion that has quietly become wrong is the thing that misleads the next session.

**Second and third occurrences arrived within the same session.** The test count drifted twice more (302 → 321 → 337) as new suites landed, and CLAUDE.md's always-loaded CLV rule carried "31% in 2025-26" — the *lower-tier* figure — where Phase 6's all-division null that season was 38.9%. An unscoped number in an always-loaded file is the same failure as a stale one.

Still one *session* rather than three, so not promoted. **Promote on the next session that finds either a stale conclusion or an unscoped figure in CLAUDE.md**, and the promoted wording should be: step 9's staleness pass covers claims and scope, not just measurements — ask of each number "is this still true, and is it labelled with what it is true *of*?"

A related catch worth recording because it came from the advisor rather than from any wrap-up step: the handoff claimed "three independent confirmations" for H1, Phase 6 and H3 — hours after this same session had correctly written that four price ladders were "one result seen from four angles". **A lesson applied at one level does not apply itself at the next**, and no checklist step would have caught it. That one is promoted to the project CLAUDE.md rather than left here, since it is about research claims and not about wrap-up.

*(Promoted this session and therefore not left here: the "step 3's real work is arguing a configuration DOWN" entry, which had a stated trigger of a third occurrence. The third occurrence arrived — eight controls against one real evaluation — so it is now in SKILL.md step 3, with the requirement that a null result enumerate what was considered rather than assert that nothing was.)*

*(Deleted, having been superseded: the entry recording that the registry seed was wrong on first reconciliation. Its stated follow-up was "if a second reconciliation also finds the total understated, promote". Two reconciliations have since run and neither found an understatement — the count moved 47 → 48 exactly as the ruling predicted — so the concern did not recur and the entry is not worth carrying.)*
