---
name: conductor
description: Use in cppfort when work must follow repo-local conductor truth under /conductor, especially to choose the next bounded compiler slice, reconcile stale track state, implement through delegated workers, or verify and close a slice without widening scope.
---

# Conductor

Conductor in this repo is the scheduler and truth authority for bounded compiler work.

## Read First

Start with the repo-local truth, in this order:
1. `conductor/tracks.md`
2. `conductor/workflow.md`
3. `conductor/setup_state.json`
4. The active track's `plan.md` and `spec.md`

## Truth Ownership

- `conductor/` is the source of truth.
- The master edits `conductor/` directly when track state is stale, missing, or overstated.
- Delegated workers own product-file edits outside `conductor/`.
- If no open bounded slice exists, create or repair one in `conductor/` and immediately assign it.

## Implement Contract

`implement` means one bounded slice, one exact corpus, one honest verification surface.

The minimum acceptable sequence is:
1. Name the slice.
2. Pin the bounded corpus.
3. Delegate product edits to one worker.
4. Verify raw diffs and real command output.
5. Update `conductor/` truth.

Do not widen scope just because adjacent debt exists.

## Worker Cast

Use the local `.kilocode` skills as a staged cast, not a swarm:
- `ir-architect`: primary writer for compiler and parser slices.
- `equivalence-verifier`: post-write verification only. No product edits.
- `code-polisher`: final writer after green verification.

Default lock order for compiler work:
`ir-architect -> equivalence-verifier -> code-polisher`

## Verification Surfaces

Prefer repo-owned verification first:
```bash
ninja -C build selfhost_rbcursive_smoke
ctest --test-dir build -R selfhost_rbcursive_smoke --output-on-failure
```

## Do Not

- Do not push track mechanics back to the user.
- Do not modify TrikeShed.
- Do not let verifier and writer co-own the same step.
- Do not mark a slice complete without a real passing verification command.
