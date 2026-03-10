# Inverse Inference Prototype

Policy: `ninja -C build conveyor` is the supported entrypoint. Direct script runs in this folder are cheats and illegal, kept only for development and debugging of the inference internals.

This folder contains tools that parse C/C++ source with libclang and perform
"inverse inference" to map AST subtrees to MLIR-like region/op templates.
The goal is to bootstrap mappings from Clang AST to `cppfront`/`cppfort`
front IRs using tree structures rather than string patterns.

## Overview

- **parse_and_infer.py**: Basic AST→regions inference (original prototype)
- **emit_mappings.py**: Emits mapping candidates per `docs/MAPPING_SPEC.md`
- **batch_emit_mappings.py**: Batch process multiple files and aggregate mappings
- **run_inference.sh**: Legacy wrapper whose logic is now folded into `cppfront_conveyor`

## Supported workflow

Use the built conveyor:

```bash
cmake -S . -B build -G Ninja
ninja -C build conveyor
```

That path will:

- build `cppfront`
- build `cppfort`
- sync the cppfront corpus
- emit reference and candidate ASTs
- score semantic loss from isomorphs
- aggregate Clang-derived semantic mappings

## Internal development

If you are modifying the inference internals, the direct scripts still exist for iteration. They are not the contract of the repo.

Run tests with:

```bash
pytest -q
```

## Mapping Schema

Emitted mappings follow the schema in `docs/MAPPING_SPEC.md`:
- **id**: Unique mapping identifier
- **ast_kind**: Clang AST cursor kind (e.g., `FunctionDecl`, `IfStmt`)
- **son_node**: Sea-of-Nodes concept (e.g., `RegionNode`, `IfNode`)
- **mlir_template**: MLIR text template with placeholders
- **semantic_signature**: Clang-derived semantic summary used for stable inference
- **grammar_fingerprint**: Normalized AST shape used to collapse corpus spelling noise
- **semantic_sections**: High-level semantic tags such as `control-flow`, `loop`, `return`
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

- Extend heuristics to correlate these Clang semantic signatures with cppfront corpus grammar
- Validate mappings against `corpus` examples
- Integrate with MLIR emitter pipeline
