# wrap-up skill — learnings

## Staging

### 2026-09-29 — a second "claim about itself" case, and this one steered a month of reading

The 08-27 pattern now in wrap-up SKILL.md step 5 (claims rather than computations) recurred. A `src/grade.py` comment said *"No cron change fixes this"* about the Friday misses, and the PROGRAMME handoff repeated it. For a month, nobody measured it. When `scripts/forward_coverage_attribution.py` did, all 86 Friday 19:00/20:00 misses were lateness, which a cron change does fix. The claim was an inference from one mechanism (the 17:30 kickoff) generalised to a whole slot. The handoff's "cron has never fired" warning had also sat unanswered for a month, because no session opened between the schedule starting and the failure.

Second occurrence of the pattern. It may belong in `/housekeeping`: a check that the "Where we are" section's open checks are dated, and that any past their date gets flagged. Not proposed as a wrap-up step.

<!-- 2026-08-27 (2nd session; step 3 argued nine things down, the hardest one flipped a sign) — pruned 2026-10-05 by /housekeeping: carried in wrap-up SKILL.md:18 (ask what the thing was searching for) and docs/PROGRAMME.md:173 (the measured_shortening_null reasoning). -->

<!-- 2026-08-27 (2nd session; a fourth check belongs in step 5) — pruned 2026-10-05 by /housekeeping: promoted into wrap-up SKILL.md step 5 (check the code's comments and docstrings too, not only the docs); recurred 2026-09-29. -->
