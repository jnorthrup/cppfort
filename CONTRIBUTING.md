# Contributing — Running BMAD tools locally

This project includes small BMAD helper scripts that extract "APEX" TODO items and optionally emit BMAD stories or create GitHub issues. This document explains how to run them locally and how the smoke-test CI job works.

## Quick start (local)

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install minimal runtime dependencies:

```bash
pip install pyyaml cryptography requests
# Or if you have a project requirements file:
# pip install -r requirements.txt
```

3. Run the extractor (example):

```bash
python3 scripts/extract_apex_swimlanes.py TODO.md --auto-promote --emit-stories --manifest manifests/apex_manifest.csv
```

Notes:
- Use `--auto-promote` to let the extractor heuristically promote unmarked checklist items to APEX swimlanes.
- By default the tool will not echo raw TODO text into created issues. Use `--echo` (opt-in) if you explicitly want that behavior.
- To create GitHub issues, provide `--create-issues --repo owner/repo` and set `GITHUB_TOKEN` in your environment.

## Making scripts runnable via `./scripts/...`

The scripts include a Python shebang. To run them directly as `./scripts/extract_apex_swimlanes.py`, make them executable:

```bash
chmod +x scripts/extract_apex_swimlanes.py scripts/decrypt_audit.py
./scripts/extract_apex_swimlanes.py TODO.md --auto-promote
```

## Encrypted local audit

- The extractor can write an encrypted local audit mapping task IDs -> original TODO text. The audit key is stored in `~/.bmad_core_audit_key` or can be provided via `BMAD_AUDIT_KEY`.
- To decrypt an audit file created under `.bmad-core/audit/`, run:

```bash
python3 scripts/decrypt_audit.py .bmad-core/audit/audit_YYYYMMDDT...json.enc
```

## Environment variables

- `BMAD_DISABLE_LOCAL_AUDIT=1` : disable writing the local encrypted audit (useful for CI runners)
- `BMAD_AUDIT_KEY` : optional base64 Fernet key (overrides `~/.bmad_core_audit_key`)
- `GITHUB_TOKEN` : required for API issue creation if `gh` CLI is not present

## CI smoke test

The repository includes a GitHub Actions workflow (`.github/workflows/extractor-smoke-test.yml`) that runs the extractor on `TODO.md` in a disposable environment to catch runtime issues (missing deps, syntax errors). Use the workflow_dispatch or run it on PRs.

If you need help, open an issue or reply on the PR created for the BMAD tooling changes.
