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

## [ ] Track: Full Corpus Transpile Validation - Match Cppfront Output
*Link: [./conductor/tracks/corpus_validation_20251230/](./conductor/tracks/corpus_validation_20251230/)*
*Status: ACTIVE* - Single-phase validation of all 189 corpus files

**Objective**: Achieve 100% transpile accuracy matching cppfront reference output

**Scope**:
- 189 corpus files (139 pure2, 50 mixed)
- Sequential processing in sorted order
- Git worktree isolation for all fixes
- Semantic loss target: <0.05 average
- Full completion: 189/189 files transpiling successfully

**Recent Progress**:
- 2026-01-01: Template argument preservation (+13 files, 63.5% → 70.4%)
  - 133/189 passing (pure2: 93/139 [66.9%], mixed: 40/50 [80.0%])
  - Fixed: Template argument capture for non-type template parameters
- 2025-12-31: Parser improvements (+120 files, 4.2% → 63.5%)
  - Fixed: == compile-time functions, @flag_enum, postfix is, named returns, access specifiers, concept keyword

**Remaining Blockers** (56 files):
1. Variadics (Ts...) parameter packs
2. Type aliases and namespace aliases
3. Some inspect/is patterns
4. UFCS edge cases

---

*Total Tracks: 6*
*Completed: 3*
*In Progress: 0*
*Active: 1*
*Planned: 1*
*Blocked: 1*
