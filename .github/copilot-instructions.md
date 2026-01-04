<!-- Copilot instructions for the Cppfort repository -->
# Cppfort — Copilot guidance (concise)

Purpose: Help an AI coding assistant be immediately productive in `cppfort` by
calling out architecture, developer workflows, and repository-specific
conventions the codebase relies on.

- **Big picture:** Cppfort is a Cpp2→C++ transpiler with two main modes: a
  traditional AST pipeline and a Sea-of-Nodes (SoN) graph-based pipeline that
  leverages an MLIR dialect (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)).
  Key concepts: Pijul CRDT graph, SoN IR, MLIR TableGen dialect, and a Global
  Code Motion scheduler.

- **Where to look first:**
  - Dialect and op/type definitions: [include/Cpp2Dialect.td](include/Cpp2Dialect.td)
  - Architecture overview: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
  - Regression test system / invocation examples: [docs/REGRESSION_TESTING.md](docs/REGRESSION_TESTING.md)
  - Language behavior/conventions: [docs/cppfront/mixed.md](docs/cppfront/mixed.md)

- **Build & test workflows (concrete commands):**
  - Build: `cmake -G Ninja -B build && ninja -C build`
  - Unit/regression tests: `ninja -C build test` or run the regression runner
    directly: `./build/tests/regression_runner <tests> <cppfront-bin> ./build/src/cppfort ./corpus`
  - Regression convenience script referenced in docs: `./scripts/run_regression.sh` (see [docs/REGRESSION_TESTING.md](docs/REGRESSION_TESTING.md)), but prefer direct invocation when script is absent.

- **Project-specific conventions and patterns:**
  - Cpp2 and Cpp1 can be mixed side-by-side in a single `.cpp2` file but **not** nested; Cpp2 sections are order-independent while Cpp1 is order-dependent ([docs/cppfront/mixed.md](docs/cppfront/mixed.md)).
  - Tests are integrity-checked via a SHA256 database; regression outputs are compared _isomorphically_ (AST/semantic equivalence rather than textual diff) — see `sha256_database.txt` and `corpus/` layout in `docs/REGRESSION_TESTING.md`.
  - MLIR dialects are TableGen-driven; generated include files follow the `Cpp2Ops*.inc` pattern described in `docs/ARCHITECTURE.md`.

- **Integration points & extension notes:**
  - MLIR bridge: bidirectional conversion between CRDT/SoN graphs and MLIR (see architecture doc); changes to dialects require regenerating TableGen outputs.
  - Regression comparison logic lives in `regression_runner` (search for `IsomorphicComparator` in tests/docs to locate the comparator class to modify when adding comparison rules).
  - Many chapter examples live under `docs/sea-of-nodes/chapter*` and include `Makefile`/`pom.xml` for example builds or demos.
  - Prototype inference tool: `tools/inference/parse_and_infer.py` parses C/C++ with libclang and emits JSON mappings (AST extents → inferred regions/blocks). Use it as a starting point to derive clang→cppfront mapping rules.

- **When making PRs / edits:**
  - Prefer small, focused changes that include a test demonstrating behavior (update the corpus or regression inputs if relevant).
  - If changing the dialect or op semantics, include regenerated TableGen outputs and a short explanation of the change's impact on the SoN→MLIR bridge.

- **Quick heuristics for code changes:**
  - If a change affects emitted C++ from SoN/MLIR, add/adjust a regression test and run `regression_runner` to prove semantic equivalence.
  - If anything touches TableGen/dialect files, run the TableGen generation step used by the CMake workflow and verify build.

If anything here is unclear or you want different emphasis (more examples, CI notes, or API-level call sites), tell me which sections to expand and I will iterate.
