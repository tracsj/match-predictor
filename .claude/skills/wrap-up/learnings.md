# wrap-up skill — learnings

## Staging

### 2026-08-17 (3rd session) — step 9's staleness pass caught two counts and one wrong conclusion
The staleness sub-step usually finds nothing. This session it found `CLAUDE.md` claiming **299 tests** against an actual 302, and "the four self-tests" against five, neither of which any other step would have surfaced — the suite was green both times, so nothing failed to draw attention to the drift.

More importantly it forced the question of whether a *conclusion* in `CLAUDE.md` was still true, not just a number. It was not: the gloss on the founding study's CLV depended on a null the session had just disproved. **The pass is worth more when read as "is every claim here still true?" rather than "have any numbers moved?"** — a number that drifts is cheap to fix, and a conclusion that has quietly become wrong is the thing that misleads the next session.

One occurrence, so left here rather than promoted. If a second session finds a stale *conclusion* rather than a stale number, promote into step 9: say explicitly that the pass covers claims, not just measurements.

*(Promoted this session and therefore not left here: the "step 3's real work is arguing a configuration DOWN" entry, which had a stated trigger of a third occurrence. The third occurrence arrived — eight controls against one real evaluation — so it is now in SKILL.md step 3, with the requirement that a null result enumerate what was considered rather than assert that nothing was.)*

*(Deleted, having been superseded: the entry recording that the registry seed was wrong on first reconciliation. Its stated follow-up was "if a second reconciliation also finds the total understated, promote". Two reconciliations have since run and neither found an understatement — the count moved 47 → 48 exactly as the ruling predicted — so the concern did not recur and the entry is not worth carrying.)*
