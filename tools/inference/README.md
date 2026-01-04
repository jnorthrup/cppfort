# Inverse Inference Prototype

This folder contains tools that parse C/C++ source with libclang and perform
"inverse inference" to map AST subtrees to MLIR-like region/op templates.
The goal is to bootstrap mappings from Clang AST to `cppfront`/`cppfort`
front IRs using tree structures rather than string patterns.

## Overview

- **parse_and_infer.py**: Basic AST→regions inference (original prototype)
- **emit_mappings.py**: Emits mapping candidates per `docs/MAPPING_SPEC.md`
- **batch_emit_mappings.py**: Batch process multiple files and aggregate mappings
- **run_inference.sh**: Wrapper script that configures libclang paths and venv

## Quick start

1. Run on a single C++ file:

```bash
./tools/inference/run_inference.sh tools/inference/emit_mappings.py \
  -i tools/inference/samples/sample_son.cpp \
  -o /tmp/mappings.json -- -std=c++20
```

2. Batch process multiple files:

```bash
python3 tools/inference/batch_emit_mappings.py \
  -i tools/inference/samples \
  -o /tmp/batch_output \
  --aggregate
```

3. Run tests:

```bash
pytest -q
```

## Mapping Schema

Emitted mappings follow the schema in `docs/MAPPING_SPEC.md`:
- **id**: Unique mapping identifier
- **ast_kind**: Clang AST cursor kind (e.g., `FunctionDecl`, `IfStmt`)
- **son_node**: Sea-of-Nodes concept (e.g., `RegionNode`, `IfNode`)
- **mlir_template**: MLIR text template with placeholders
- **confidence**: Heuristic confidence score (0.0-1.0)
- **examples**: Representative input/output snippets

## Known Limitations

- **cpp2 files**: `.cpp2` files contain cppfront syntax that Clang cannot parse
  directly. To process `.cpp2` files, first transpile them to `.cpp` with
  `cppfront`, then run inference on the generated C++.
  
- **Standard C++ focus**: Current mappings target standard C++ constructs.
  Extend heuristics in `emit_mappings.py` to recognize cpp2-specific patterns
  (UFCS, contracts, `inout` parameters, etc.).

## Next steps

- Extend heuristics to handle cpp2 semantics (after cppfront transpilation)
- Validate mappings against `corpus` examples
- Integrate with MLIR emitter pipeline
