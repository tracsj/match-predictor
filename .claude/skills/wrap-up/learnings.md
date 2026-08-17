# wrap-up skill — learnings

## Staging

### 2026-08-17 — the registry step caught its own seed being wrong on first run
Step 3 was seeded at ~38 configurations from `PREREGISTRATION.md` plus the confidence analysis. Enumerating properly at wrap-up gave **47** — the tier-shift sweep was 11 values where 7 were disclosed, and the tier-2 arms and feature-set variants had never been counted at all.

Worth keeping in Staging rather than promoting: it is one occurrence, and the step already works. But if a second reconciliation also finds the running total understated, the lesson is that counts must be incremented *at the moment a configuration is scored*, not reconstructed at session close — reconstruction depends on memory of a long session, which is exactly what fails.

*(Promoted this session, so not left here: step 6's 150-line file threshold produced pure noise on a Python research repo — 20+ files exceed it and most are cohesive. SKILL.md now says flag the top 3 and only on growth.)*
