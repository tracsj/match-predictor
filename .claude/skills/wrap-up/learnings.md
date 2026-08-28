# wrap-up skill — learnings

## Staging

### 2026-08-27 (2nd session) — step 3 argued nine things down, and the hardest one flipped a sign

The reconciliation's real work was again subtraction, and one candidate was genuinely hard rather than obviously a control. `measured_shortening_null` changed the forward ledger's CLV reading from p 0.44 to a nominal p 0.031 — a sign flip, which is what an evaluation looks like from the outside. **The test that settled it was not "did the number move?" but "was an alternative tried and discarded?"** It was not: the bet population was fixed by a committed workflow before the null existed, and the null is a property of the market rather than of any model. What changed was the yardstick, not the search.

Worth carrying because the existing step-3 guidance handles controls well and says nothing about this case. **A change of yardstick can move a headline further than a real configuration would and still not widen the search.** If it recurs, promote as: ask what was *searched over*, not what changed.

One-off for now. Second occurrence promotes.

### 2026-08-27 (2nd session) — a fourth check belongs in step 5, and the session found it by accident

The suite was green all evening and three real faults sat underneath it: a workflow that had never registered with GitHub Actions, a `clv_report` null the docs had corrected ten days earlier and the code had not, and a `betting.py` docstring advertising a Kelly simulation that has never existed. **Not one of them could fail a test**, because each was a claim rather than a behaviour.

No new step proposed — step 9's staleness pass is the right home and has been widened to say "check the code too, not only the docs". Recorded here because the pattern is that **the things this repo gets wrong are increasingly claims about itself rather than computations**, and the wrap-up's test-suite step cannot reach any of them.

*(Promoted this session and therefore not left here: the "step 3's real work is arguing a configuration DOWN" entry, which had a stated trigger of a third occurrence. The third occurrence arrived — eight controls against one real evaluation — so it is now in SKILL.md step 3, with the requirement that a null result enumerate what was considered rather than assert that nothing was.)*

*(Deleted, having been superseded: the entry recording that the registry seed was wrong on first reconciliation. Its stated follow-up was "if a second reconciliation also finds the total understated, promote". Two reconciliations have since run and neither found an understatement — the count moved 47 → 48 exactly as the ruling predicted — so the concern did not recur and the entry is not worth carrying.)*
