# Public Snapshot Manifest

## Snapshot identity

- Private source commit: `a308375`
- Export type: clean file snapshot with a new Git root
- Public history inherited from private repository: no
- Mobile client included: no
- Public replay datasets included: no
- Private phone datasets included: no

## Included

- FastAPI application and responsive Web UI
- Agent runtime, context/state, provider compatibility and run audit
- High/Fast user selection, visible-output streaming and explicit user-owned fallback control
- Diagnostic and Exploration product state machines
- Two zero-model showcase chains: washing-machine diagnosis and four-step light exploration
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

The public snapshot is designed to start locally and demonstrate the Web product, account isolation, two zero-wait evidence chains, protocol creation and optional model-backed diagnostic/Exploration flows. Users must supply their own compatible model credentials only for real model operations. Third-party dataset-backed replay cards are intentionally unavailable; the bundled showcase records are server-generated, explicitly non-physical fixtures.
