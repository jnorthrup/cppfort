# Workflow

## Main Todo (Recap)

Implement the next incomplete Conductor track by completing a small, testable subset of tasks, updating the track plan as you go, and leaving a clear handoff note for the next session.

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
