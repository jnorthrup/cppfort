# cppfort external loop

Operate on one bounded compiler slice at a time.

## Entry Point

Use:

```bash
python3 /Users/jim/work/cppfort/train.py
```

Optional bounded flags:

```bash
python3 /Users/jim/work/cppfort/train.py --limit 25 --skip-mappings
```

## Mutable Surface

Change one canonical owner only:

- grammar and parsing: `include/parser_grammar.hpp`, `include/parser/*`, parser sources
- semantic ownership: `include/ast.hpp`, semantic analyzer sources
- emission/codegen: emitter/codegen sources
- conveyor contract only when the failure is in the conveyor itself

Do not widen the hole.

## Fixed Surface

- `train.py` is the measurement harness
- do not use upstream cppfront as the transpiler under test
- do not replace candidate output with oracle artifacts

## Keep Rule

Keep a change only if one of these improves honestly:

- `metrics.cppfort_transpiles_on_primary_corpus`
- `metrics.cppfort_failures_on_primary_corpus` decreases
- `metrics.average_combined_semantic_loss` decreases
- same score, simpler code

## One-Slice Loop

1. Read the current failure surface.
2. Pick one bounded owner.
3. Edit only that slice.
4. Run `python3 train.py`.
5. Read `runs/<run-id>/result.json`.
6. Keep or rollback.

## Suggested Prompt

```text
Read program.md. Pick one bounded cppfort ownership slice, modify only that slice, run python3 train.py, inspect runs/<run-id>/result.json, and keep or rollback.
```
