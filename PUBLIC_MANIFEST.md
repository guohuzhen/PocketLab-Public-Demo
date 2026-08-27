# Public Snapshot Manifest

## Snapshot identity

- Private source commit: `b2366ae`
- Export type: clean file snapshot with a new Git root
- Public history inherited from private repository: no
- Mobile client included: no
- Public replay datasets included: no
- Private phone datasets included: no

## Included

- FastAPI application and responsive Web UI
- Agent runtime, context/state, provider compatibility and run audit
- Diagnostic and Exploration product state machines
- Deterministic sensor analyzers and phyphox bridge
- Local account, persistence and evidence workbench
- Dependency lockfile and environment template
- Git safety scanner and pre-commit hook
- Public-only README, quickstart, security notice and smoke test

## Excluded

- Original `.git` directory, branches, tags and pull-request history
- `.env.local`, real provider credentials and model endpoints
- SQLite databases, cookies, sessions, logs and traces
- `datasets/phone` and other user-collected measurements
- `datasets/public` replay packs, including license-unresolved local-only packs
- Holdout Evals, adversarial prompts, scoring thresholds and raw model outputs
- Internal Harness implementations and development loop notes
- HarmonyOS client skeleton
- Competition submission forms, scripts and personal contact details

## Functional boundary

The public snapshot is designed to start locally and demonstrate the Web product, account isolation, protocol creation and model-backed diagnostic/Exploration flows. Users must supply their own compatible model credentials. Dataset-backed public replay cards are intentionally unavailable in this snapshot.
