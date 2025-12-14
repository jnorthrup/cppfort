# Mapping Task: Clang AST ↔ Normalized MLIR Regions

Goal
----
Capture Clang AST graph isomorphs of normalized MLIR base-dialect region expressions and produce validated mapping artifacts.

Deliverables
------------
- JSON/YAML mapping schema for Clang AST extents → SoN/MLIR region templates (include confidence and examples).
- Extractor tool that ingests Clang AST outputs and normalized MLIR (base dialect) region expressions and emits mappings.
- Validation tests using `corpus/inputs` and selected `docs/sea-of-nodes` chapter examples.
- Integration hooks to feed mapping artifacts into the MLIR emitter.

Next steps
----------
1. Draft mapping schema (`docs/MAPPING_SPEC.md`).
2. Extend `tools/inference/parse_and_infer.py` to emit candidate mappings.
3. Add unit tests and a small CI smoke job.

Notes
-----
This task is tracked as TODO id `26` in the project's task list and is intended to run without bypassing or faking test results.
