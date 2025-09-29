# Stage1: regression analysis helpers

This folder contains small tooling to help Stage1 ingest historical
regression results so inductive learning and pattern extraction can
begin. The goal is to turn the plain-text `regression-tests/regression_log.txt`
into structured data and then extract patterns that explain failures.

Files
- `parse_regression.py` - small dependency-free parser producing
  `regression_summary.json` and `regression_summary.csv` in this folder.

Suggested Stage1 workflow
1. Run the parser to convert the regression log into structured data.
   ```
   ./tools/stage1/parse_regression.py regression-tests/regression_log.txt
   ```
2. Build features per-test: error types, file+line ranges, AST node kinds
   (link ASTs from `src/stage0/ast.h` to failing tests by parsing the
   `.cpp2` sources and mapping error locations).
3. Use simple clustering or rule extraction (decision trees, frequent
   substrings) to propose candidate fixes or parser/emitter rule changes.
4. Create small candidate patches to `src/stage0` components and re-run
   regression to validate.

Notes
- This is intentionally small. Stage1 should extend the parser to capture
  richer context (surrounding source lines, token streams) and to link
  failures to AST structures.
