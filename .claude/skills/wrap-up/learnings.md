# wrap-up skill — learnings

## Staging

### 2026-08-17 (2nd session) — step 3's real work was arguing a configuration DOWN, not up
The step is written to catch an undercount. This session the honest answer was that nothing counted, and getting there took more care than adding an entry would have: a dry run trained three seeds, and a grading dry run produced CLV and ROI across four price columns. Both look exactly like evaluations in a log.

What settled it was asking what each thing was *searching for*. The retrain was the same pre-registered configuration on unplayed fixtures with nothing scored; the grading run's "model" was the de-vigged closing price itself, a control that must place zero bets. Neither searched for edge.

Recorded as three named items in `PROGRAMME.md` rather than a bare "nothing evaluated", because "nothing" is indistinguishable from having skipped the step. **If a third session also has to argue something down, promote this into SKILL.md step 3** — the step should say that a null result must enumerate what was considered and why it does not count.

*(Promoted this session, so not left here: step 2's allow-vs-deny false positive, and `FORWARD_LEDGER.md` missing from step 7's skip list. Both are now fixed in SKILL.md.)*

### 2026-08-17 — the registry step caught its own seed being wrong on first run
Step 3 was seeded at ~38 configurations from `PREREGISTRATION.md` plus the confidence analysis. Enumerating properly at wrap-up gave **47** — the tier-shift sweep was 11 values where 7 were disclosed, and the tier-2 arms and feature-set variants had never been counted at all.

Worth keeping in Staging rather than promoting: it is one occurrence, and the step already works. But if a second reconciliation also finds the running total understated, the lesson is that counts must be incremented *at the moment a configuration is scored*, not reconstructed at session close — reconstruction depends on memory of a long session, which is exactly what fails.

*(Promoted this session, so not left here: step 6's 150-line file threshold produced pure noise on a Python research repo — 20+ files exceed it and most are cohesive. SKILL.md now says flag the top 3 and only on growth.)*
