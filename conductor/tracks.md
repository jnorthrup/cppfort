# Project Tracks

This file tracks all major tracks for the project. Each track has its own detailed plan in its respective folder.

---

## [x] Track: Establish Core Cpp2 to MLIR Front-IR Conversion and Sea of Nodes Dialect Integration
*Link: [./conductor/tracks/cpp2_mlir_son_20251222/](./conductor/tracks/cpp2_mlir_son_20251222/)*
*Status: COMPLETE*

---

## [x] Track: SCCP pass implementation
*Link: [./conductor/tracks/sccp_20251227/](./conductor/tracks/sccp_20251227/)*
*Status: COMPLETE* - All 5 phases verified, 3 verification gaps resolved
- Code coverage: 72.9% (exceeds >20% requirement by 3.6x)
- Debug logging: Full LLVM_DEBUG support implemented
- All 7 SCCP tests passing
- Checkpoints: Phase 4 (30c150a), Phase 5 (c1b08c8)
- Gap resolutions: 2b1021d, 5df7fe3, 8554cf1, 82305f0

---

## [x] Track: Markdown comments with CAS-linked module stubs
*Link: [./conductor/tracks/markdown_cas_20251227/](./conductor/tracks/markdown_cas_20251227/)*
*Status: COMPLETE - All phases verified, comment-wrapped syntax implemented*

**Implementation Summary:**
- **Lexer:** MARKDOWN_BLOCK token with comment-wrapped syntax (`/*```...```*/`)
- **SHA256:** Trim-and-concatenate algorithm with known test vectors
- **AST:** MarkdownBlockAttr metadata structure attached to declarations
- **Code Generation:** Empty C++20 module stubs with SHA256 constants
- **Testing:** 5 test suites, 31 total tests, all passing
- **Quality:** Edge cases covered (empty blocks, Unicode, special characters, nested code)
- **Verification:** Independent gap analysis confirms 8/8 requirements met

---

## [ ] Track: Semantic AST Enhancements (Escape Analysis, Borrowing, External Memory, Channels)
*Link: [./conductor/tracks/semantic_ast_20251230/](./conductor/tracks/semantic_ast_20251230/)*
*Status: PLANNED* - 6-phase roadmap (15-18 days)

**Objectives**:
- Escape analysis framework (track value lifetimes and escape points)
- Borrowing and ownership tracking (Rust-like semantics)
- External memory pipeline integration (GPU/DMA transfers, lifecycle optimization)
- Channelized concurrency integration (ownership through channels, data race detection)
- Unified semantic representation (SemanticInfo attached to all AST nodes)

**Target Metrics**:
- Parameter semantics: 0% → 100%
- Escape analysis coverage: 0% → 100%
- Average corpus semantic loss: 1.0 → <0.15

---

## [ ] Track: Regression Test Corpus Semantic Preservation
*Link: [./conductor/tracks/regression_corpus_20251230/](./conductor/tracks/regression_corpus_20251230/)*
*Status: BLOCKED* - Infrastructure complete, transpiler fixes required

**Blockers**:
1. P0: Parameter semantics lost (inout → by-value instead of by-reference)
2. P1: Mixed-mode C++1 syntax support (50/189 tests blocked)
3. P2: Semantic loss scoring accuracy

**Current Results**:
- pure2-hello.cpp2: Transpiles but semantic loss = 1.0 (max)
- Corpus infrastructure: 1.4M isomorphs, 13.5K unique patterns, 100% MLIR coverage
- Test status: pure2 works, mixed fails

---

## [~] Track: Full Corpus Transpile Validation - Match Cppfront Output
*Link: [./conductor/tracks/corpus_validation_20251230/](./conductor/tracks/corpus_validation_20251230/)*
*Status: ACTIVE* - Single-phase validation of all 189 corpus files

**Objective**: Achieve 100% transpile accuracy matching cppfront reference output

**Scope**:
- 189 corpus files (139 pure2, 50 mixed)
- Sequential processing in sorted order
- Git worktree isolation for all fixes
- Semantic loss target: <0.05 average
- Full completion: 189/189 files transpiling successfully

**Current Baseline** (2026-01-02):
- 21/189 passing (11.1%) - ACCURATE baseline after test framework fix
- Previous 163/189 (86.2%) claim was based on transpilation success only, NOT full compile+run
- Test framework fix: Added `-I third_party/cppfront/include` to compilation

**Recent Progress**:
- 2026-01-02: Test framework fix (+10 tests, 11 → 21 passing)
  - Added cpp2util.h include path to compilation
  - New passes: mixed-bounds-check, mixed-hello, pure2-hello, pure2-break-continue, etc.
- 2026-01-02: For-loop disambiguation (transpilation success)
- 2026-01-02: Type alias support (transpilation success)
- 2026-01-02: C++1 syntax detection improvements (transpilation success)
- 2026-01-01: Loop initializer syntax (transpilation success)
- 2026-01-01: Template argument preservation (transpilation success)
- 2025-12-31: Parser improvements (transpilation success)

**Remaining Blockers** (147 non-error tests failing):
1. Pattern matching (`inspect`, `is`, `as` operators) - ~20 tests
2. Function expressions (unbraced syntax) - ~9 tests
3. Type system features (advanced types, constraints) - ~17 tests
4. Contract assertions - ~4 tests
5. String interpolation (`$"..."$`) - ~3 tests
6. UFCS in complex contexts - ~6 tests
7. Various language features (autodiff, bounds, parameters) - ~88 tests

---

*Total Tracks: 6*
*Completed: 3*
*In Progress: 0*
*Active: 1*
*Planned: 1*
*Blocked: 1*
