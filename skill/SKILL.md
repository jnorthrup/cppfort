---
name: conductor
description: Use in cppfort when work must follow repo-local conductor truth under /conductor, especially to choose the next bounded compiler slice, reconcile stale track state, implement through delegated workers, or verify and close a slice without widening scope. Also use when the user mentions conductor, track, plan, spec, implement, next slice, bounded corpus, delegated worker routing, or status.
---

# Conductor

Conductor in this repo is not a generic planner. It is the scheduler and truth authority for bounded compiler work.

## Read First

Start with the repo-local truth, in this order:

1. `conductor/tracks.md`
2. `conductor/workflow.md`
3. `conductor/setup_state.json`
4. The active track's `plan.md` and `spec.md`

If the active track says to consult TrikeShed, treat `/Users/jim/work/TrikeShed` as read-only reference only.

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

## Dining-Philosophers Lock

Treat the workflow as a lock order:

- Conductor holds the truth fork.
- At most one worker may hold the product-code fork for a given corpus.
- Verification starts only after the writer releases that corpus.
- Final polish starts only after verification is green.
- Never let two writers co-edit the same files.

If a worker dies before repo edits, reopen the same slice. Do not widen scope and do not narrate success.

## Worker Cast

Use the local `.kilocode` skills as a staged cast, not a swarm:

- `algebraic-optimizer`: read-only proof advisor for rewrite or IR algebra. Skip it unless the slice is really about semantic rewrite rules.
- `ir-architect`: primary writer for compiler and parser slices.
- `equivalence-verifier`: post-write verification only. No product edits.
- `code-polisher`: final writer after green verification.

Default lock order for compiler work:

`ir-architect -> equivalence-verifier -> code-polisher`

For the current selfhost parser work, `algebraic-optimizer` is usually unnecessary.

## Current cppfort Focus

The active selfhost dogfood path lives in:

- `src/selfhost/rbcursive.cpp2`
- `tests/selfhost_rbcursive_smoke.cpp`
- `src/selfhost/CMakeLists.txt`

The current untouched bounded slice is the alpha surface:

- `series alpha (x) => expr`

Keep that slice bounded to the selfhost parser and smoke corpus unless the track truth explicitly expands it.

## Verification Surfaces

Prefer repo-owned verification first:

```bash
ninja -C build selfhost_rbcursive_smoke
ctest --test-dir build -R selfhost_rbcursive_smoke --output-on-failure
```

If CMake regeneration is flaky but the track allows fallback, use the direct bridge:

```bash
/Users/jim/.local/bin/cppfront -p -q -o build/selfhost/rbcursive.cpp src/selfhost/rbcursive.cpp2
/usr/bin/clang++ -std=c++20 -U__cpp_lib_modules -DCPPFORT_SOURCE_DIR=\"/Users/jim/work/cppfort\" -I/Users/jim/work/cppfort/build/selfhost -I/Users/jim/work/cppfront/include tests/selfhost_rbcursive_smoke.cpp -o /tmp/selfhost_rbcursive_smoke_manual
/tmp/selfhost_rbcursive_smoke_manual
```

Always record which route actually passed.

## Runtime Handling

Delegated execution is required for product-file edits in normal operation.

- If a worker runtime is broken, say exactly how it failed.
- Distinguish runtime failure from build failure from product failure.
- Do not pretend a blocked worker is equivalent to a completed slice.

## Do Not

- Do not push track mechanics back to the user.
- Do not modify TrikeShed.
- Do not let verifier and writer co-own the same step.
- Do not mark a slice complete without a real passing verification command.
- Do not treat broad rediscovery as progress after the slice is already named.
