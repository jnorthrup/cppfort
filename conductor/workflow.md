# Workflow

## Main Todo (Recap)

Implement the next incomplete Conductor track by completing a small, testable subset of tasks, updating the track plan as you go, and leaving a clear handoff note for the next session.

## One-Slice Loop Principles

Based on [`program.md`](../program.md:1), the workflow follows a bounded one-slice loop:

1. **Read the current failure surface** - Identify the specific issue or gap
2. **Pick one bounded owner** - Select a single canonical owner (grammar, semantic, emission)
3. **Edit only that slice** - Make changes to one bounded compiler slice at a time
4. **Run validation** - Execute `python3 train.py` or targeted tests
5. **Inspect results** - Read `runs/<run-id>/result.json` or test output
6. **Keep or rollback** - Apply the "Keep Rule" to decide whether to keep changes

### Keep Rule
Keep a change only if one of these improves honestly:
- `metrics.cppfort_transpiles_on_primary_corpus` increases
- `metrics.cppfort_failures_on_primary_corpus` decreases
- `metrics.average_combined_semantic_loss` decreases
- Same score, simpler code

## External Loop Operations

The external loop operates on one bounded compiler slice at a time:

**Mutable Surface** (change one canonical owner only):
- Grammar and parsing: `include/parser_grammar.hpp`, `include/parser/*`, parser sources
- Semantic ownership: `include/ast.hpp`, semantic analyzer sources
- Emission/codegen: emitter/codegen sources
- Conveyor contract only when the failure is in the conveyor itself

**I/O Strategy**:
- **Between build stages**: Use stdio for clarity (not channelized reactor)
- **Large intermediate representations**: Use memory-mapped I/O only when necessary
- **Rationale**: Simplicity over marginal performance gains in a compiler pipeline

**Fixed Surface** (do not change):
- `train.py` is the measurement harness
- Do not use upstream cppfront as the transpiler under test
- Do not replace candidate output with oracle artifacts
- cppfront is a benchmark/validator only, not a build dependency

## Task Workflow

1. Read the selected track `plan.md` and `spec.md`.
2. Reset any stale `[~]` tasks back to `[ ]` before starting.
3. Execute tasks one at a time:
   - mark `[ ]` -> `[~]`
   - implement code/docs/tests for that task
   - run relevant validation
   - mark `[~]` -> `[x]`
4. Stop after a small batch (recommended 3-5 tasks) or if blocked.
5. Add/update a short recap in the track `plan.md` indicating the next main todo.
6. Update `conductor/tracks.md` summary/status notes if progress materially changed.

## Validation Guidance

- Prefer targeted tests first, then broader regression runs if the change touches shared code.
- Record command(s) run and pass/fail outcome in the track notes or recap.

## Blocking Rule

If track files or prerequisites are missing, create minimal stubs (`plan.md`, `spec.md`) before attempting implementation.
