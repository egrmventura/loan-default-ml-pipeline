# Agent Index

Available agents for this project. Invoke by name or describe the task — Claude Code will route to the appropriate agent.

---

## Project-Specific Agents

| Agent | Invoke when... |
|---|---|
| `ml-tester` | Before any commit, after modifying source files, or any time you want a full green-light check (lint + tests + smoke test) |
| `data-validator` | After rebuilding `loans_featured.parquet` — confirms zero nulls, correct class balance, and feature ranges before training |
| `pipeline-rerun` | Starting a new experiment — orchestrates full rebuild → retrain → evaluate → log in one pass |
| `pr-ready` | Before opening or merging a PR — full checklist including no data files, no secrets, docs current |
| `feature-auditor` | After modifying `build_features.py` — catches leakage, encoding errors, bad null fills before they corrupt training |
| `experiment-logger` | Immediately after running `evaluate_model()` — formats and appends results to `docs/experiment_log.md` |
| `schema-guard` | When pulling a new dataset or switching data sources — validates column compatibility before the pipeline runs |
| `doc-syncer` | End of session or before a PR — audits and fixes gaps between docs and actual code state |

---

## Reusable Template Agents

| Agent | Invoke when... |
|---|---|
| `ml-code-reviewer` | Reviewing any ML code change for correctness bugs — leakage, train/test contamination, encoding errors, evaluation mistakes |
| `session-handoff` | Ending a session where meaningful progress was made — generates a pickup-ready handoff doc in `docs/handoff/` |

---

## Recommended Sequences

**Before every commit:**
`ml-tester` → fix anything → commit

**Before every PR:**
`ml-tester` → `pr-ready` → fix anything → push

**Starting a new experiment:**
`data-validator` → `pipeline-rerun` → `experiment-logger`

**After modifying the feature pipeline:**
`feature-auditor` → `ml-tester` → `data-validator`

**End of session:**
`doc-syncer` → `session-handoff`
