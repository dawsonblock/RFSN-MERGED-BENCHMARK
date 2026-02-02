# Branch Protection (Required)

This repo depends on *governed autonomy*:
- planners/LLMs may propose
- gates select
- the governed executor is the only path to effects

To prevent regressions, `main` MUST be protected.

## Required GitHub settings (Settings → Branches → Branch protection rules)

Branch name pattern: `main`

Enable:
- Require a pull request before merging
  - Require approvals: 1+
- Require status checks to pass before merging
  - Require branches to be up to date before merging
  - Required checks:
    - CI / lint
    - CI / compile
    - CI / security-invariants
    - CI / tests (optional)
- Require conversation resolution before merging

Recommended:
- Restrict who can push to matching branches (maintainers only)
- Require signed commits
- Disallow force-pushes

## Why
These rules enforce that:
- SimulationGate remains advisory-only
- LLM patch flow stays APPLY_PATCH
- No bypass of the governed execution spine
