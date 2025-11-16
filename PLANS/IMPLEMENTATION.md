# Implementation changes aligned with ARCHITECTURE.md

Summary of updates completed to follow the architecture goals:

- Revived coding standards with `.clang-format` configuration (LLVM base, 4-space indent, C++20).
- Added a minimal `cpp2_cas` helper (`src/stage0/cpp2_cas.{h,cpp}`) that rewrites ```cpp2 blocks into `// CAS:<id>` placeholders (deterministic fallback). Added `test_cpp2_cas` to validate behavior.
- Added a `GraphMatcher` placeholder (`src/stage0/graph_matcher.h`) to house future graph matcher implementation, with a simple substring-matching test (`test_graph_matcher_stub`).
- Introduced a `scripts/audit_regex_usage.sh` script to help audit regex usage and track migration to graph-based matching.
- Audited repository for `#include <regex>` and `PatternType::Regex` usages; migrated `src/stage0/complete_pattern_engine.h` to remove an unnecessary `<regex>` include.
- Added a minimal JSON scanner test (`test_json_scanner_simple`) using the `JsonScanner` header implementation.

Next steps:

1. Replace `std::hash` CAS fallback with a BLAKE-based hash (e.g., BLAKE3) and add a canonicalization step.
2. Implement `GraphMatcher` using `pijul_parameter_graph` and `pijul::Graph` primitives; add round-trip tests for critical transforms.
3. Begin migrating `PatternType::Regex` usages (e.g., in `rbcursive.cpp`) to `GraphMatcher`, with tests for equivalence.
4. Fix build problems in several files that include missing headers (notably MLIR includes) to allow full `stage0_lib` builds and regression test runs.
5. Add CI hooks to run `scripts/audit_regex_usage.sh` and the new test binaries in a lightweight test step.

If you'd like, I can proceed to implement any of the next steps. Please tell me which item to tackle first.
