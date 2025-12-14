# Mapping Spec: Clang AST ↔ MLIR Region Mappings (draft)

Purpose
-------
Define a minimal, actionable schema to record mapping candidates between Clang AST subtrees (extents), Sea-of-Nodes (SoN) concepts, and normalized MLIR base-dialect region/op templates. This enables n-way mapping, validation, and a reproducible emitter pipeline.

Schema (JSON)
-------------
Each mapping is a JSON object with the following fields:

- **id**: string — unique mapping id (e.g., `func_decl_to_region`).
- **source_sample**: { file: string, span: { start: {line,col}, end: {line,col} } } — Clang AST extent example.
- **ast_kind**: string — canonical Clang AST cursor kind (e.g., `FunctionDecl`, `IfStmt`).
- **son_node**: string|null — SoN node type if known (e.g., `IfNode`, `RegionNode`).
- **mlir_template**: string — MLIR text template using placeholders, e.g.: `cpp2.if $cond { ... } else { ... }` or `cpp2.region(%$inputs) : type($results)`.
- **pattern**: string — high-level description or regex to recognize the source shape.
- **confidence**: number — heuristic confidence (0.0-1.0) assigned by extractor.
- **examples**: [ { input: string, output: string } ] — representative input/output snippets.
- **notes**: string — rationale, edge-cases, or required preconditions.

Example mapping
---------------
{
  "id": "ifstmt_to_cpp2_if",
  "source_sample": { "file": "example.cpp2", "span": { "start": {"line": 10, "col": 1}, "end": {"line": 14, "col": 1} } },
  "ast_kind": "IfStmt",
  "son_node": "IfNode",
  "mlir_template": "cpp2.if %cond { %then_region } else { %else_region }",
  "pattern": "IfStmt with then/else compound statements",
  "confidence": 0.95,
  "examples": [ { "input": "if (x>0) x=1; else x=2;", "output": "cpp2.if %cond { ... } else { ... }" } ],
  "notes": "Requires splitting compound statements into regions; map sub-exprs recursively."
}

Usage
-----
- Extract: tools/inference will emit candidate mappings as JSON documents following this schema.
- Validate: run mapping validation tests (see `tools/inference/tests`) comparing generated MLIR snippets against expected patterns and `corpus` examples.
- Integrate: mapping artifacts feed the MLIR emitter to drive SoN→MLIR lowering or AST→MLIR heuristics.

Next steps (short)
------------------
1. Add schema validation (JSON Schema or pydantic) and a CLI to emit sample mappings.
2. Extend `tools/inference/parse_and_infer.py` to emit mapping candidates in this format.
3. Add unit tests (candidate filtering, confidence thresholds) and a small CI smoke job.
