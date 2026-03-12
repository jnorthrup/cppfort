# Project Tracks

This file tracks all major tracks for the project. Each track has its own detailed plan in its respective folder.

## Conductor Recap / Main Todo

- Main todo: Execute the next incomplete track with a small, validated batch and keep `plan.md` as the source of session state.
- Immediate prerequisite (now stubbed): Ensure every track listed below has `spec.md` and `plan.md` under `conductor/tracks/<track_id>/`.
- Next target track by order: `Semantic AST Enhancements (Escape Analysis, Borrowing, External Memory, Channels)`.

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

## [~] Track: Semantic AST Enhancements (Escape Analysis, Borrowing, External Memory, Channels)
*Link: [./conductor/tracks/semantic_ast_20251230/](./conductor/tracks/semantic_ast_20251230/)*
*Status: ACTIVE* - Gap-fill semantic metadata wiring + tests in progress

**Objectives**:
- Escape analysis framework (track value lifetimes and escape points)
- Borrowing and ownership tracking (Rust-like semantics)
- External memory pipeline integration (GPU/DMA transfers, lifecycle optimization)
- Channelized concurrency integration (ownership through channels, data race detection)
- Unified semantic representation (SemanticInfo attached to all AST nodes)

**Immediate Session Focus**:
- Escape/borrow traversal skeleton pass wiring (`analyze_escape_and_borrow`)
- Placeholder metadata coverage tests (non-invasive)
- Next: Phase 4 borrow/ownership placeholder events and regression coverage

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
- 2026-01-01: Loop initializer syntax (+1 file, 70.4% → 70.9%)
  - 134/189 passing (pure2: 93/139 [66.9%], mixed: 41/50 [82.0%])
  - Implemented: `(copy i:=0)` loop initializer syntax for while/for loops
- 2026-01-01: Template argument preservation (+13 files, 63.5% → 70.4%)
  - Fixed: Template argument capture for non-type template parameters
- 2025-12-31: Parser improvements (+120 files, 4.2% → 63.5%)
  - Fixed: == compile-time functions, @flag_enum, postfix is, named returns, access specifiers, concept keyword

**Remaining Blockers** (55 files):
1. Variadics (Ts...) parameter packs (~10 files)
2. Type aliases and namespace aliases (~8 files)
3. Unary + operator, named returns with defaults (~5 files)
4. UFCS edge cases, inspect/is patterns (~10+ files)

---

## [ ] Track: Clang Back-Annotation of Generated C++ Fragments into Cpp2 AST
*Link: [./conductor/tracks/clang_back_annotation_20260225/](./conductor/tracks/clang_back_annotation_20260225/)*
*Status: PLANNED* - Metadata-only semantic oracle integration (tagging + mapping + Clang extraction)

**Objective**: Use Clang to analyze generated C++ fragments and back-annotate the originating Cpp2 AST with tagged semantic facts and provenance.

**Scope (Initial)**:
- Stable fragment tagging (`cpp2_node_id`) in generated C++
- Mapping table emission (ID -> generated source range / fragment kind)
- Narrow Clang extraction prototype (declarations, parameters, call arguments)
- Back-annotation merge into `SemanticInfo` without changing codegen behavior

**Design Constraints**:
1. Deterministic IDs and mapping stability across formatting changes
2. Additive/advisory metadata first (no silent overwrites of native semantic analyzer results)
3. Explicit handling of implicit Clang nodes, duplicates, and ambiguous mappings

---

*Total Tracks: 13*
*Completed: 8*
*In Progress: 2*
*Active: 1*
*Suspended: 0*
*Planned: 1*
*Blocked: 2*
*New: 0*

---

## [-] Track: Java Memory Model Integration for Cpp2 SON Dialect
*Link: [./conductor/tracks/son_jmm_integration_20260108/](./conductor/tracks/son_jmm_integration_20260108/)*
*Status: BLOCKED (2026-01-10)* - Phase 1 implementation complete, testing blocked by SON dialect disabled in build (LLVM 21 FieldParser issue)

**Completed:**
- JMM attributes defined in Cpp2SONDialect.td
- JMM metadata attached to LoadOp, StoreOp, NewOp, ConstructorEndOp, SendOp, RecvOp, SpawnOp, AwaitOp
- JMM constraint verification implemented (Cpp2SONJMMVerification.cpp)

**Blocker:** Cpp2SONDialect.cpp disabled in src/CMakeLists.txt - requires LLVM 21 FieldParser fix to enable testing

---


## [-] Track: Fix Compile-Time Memory Leak in Spirit Combinators
*Link: [./conductor/tracks/spirit_parser_compilefix_20250109/](./conductor/tracks/spirit_parser_compilefix_20250109/)*
*Status: SUSPENDED (2026-01-09) - Diverges from closing existing test coverage and loss reduction work*

**Progress at suspension:**
- Phase 1: Refactor rules.hpp - COMPLETE
- Phase 2: Parser Integration - COMPLETE
- Phase 3: Regression Corpus Validation - Partial (3/7 tasks)
- Phase 4: Cleanup - Not started

---

## [ ] Track: Annotation-Based Semantic Actions for Parser
*Link: [./conductor/tracks/semantic_actions_20260109/](./conductor/tracks/semantic_actions_20260109/)*
*Status: PLANNED - Not started*

**Objective:** Add annotation-based semantic idioms to existing parser (Spirit combinators + Pratt parser for 17 precedence levels) mapping to Clang AST

**Phases:**
1. Annotation infrastructure (`with_node`, `with_binary`, `ast_node`)
2. Annotate grammar rules (statements, declarations)
3. Annotate Pratt parser expressions (17 precedence levels)
4. Template and type system annotations
5. Corpus validation (target: 98.9% pass, ≤0.124 loss)

---

## [~] Track: TrikeShed Surface Restart
*Link: [./conductor/tracks/trikeshed_surface_restart_20260311/](./conductor/tracks/trikeshed_surface_restart_20260311/)*
*Status: ACTIVE* - restart cppfort’s TrikeShed migration from the real transpiler repo and its working parser/emitter harnesses

**Purpose:** Treat `../TrikeShed` as the semantic/source-text reference and land the smallest real parser/emitter slices in cppfort, starting from the beginning with harness-backed surface deltas instead of speculative sibling-repo narration.

**Immediate Focus:**
- External signal only: `/Users/jim/work/TrikeShed/conductor/grok_share_bGVnYWN5_21edd44f-9e25-434b-9bcb-2d036feee2dc.md`
- External spec input only: `/Users/jim/work/TrikeShed/conductor/tracks/cpp2-surface-transition_20260311/expanded_cpp2_spec.md`
- Local truth owner: `grammar/cpp2.ebnf`, `src/parser.cpp`, `src/emitter.cpp`, and parser tests under `tests/`
- Verified slices:
  - multi-expression subscript syntax now parses and emits for `coords[1.0, 2.0]` through existing slim parser/emitter harnesses
  - canonical grammar truth and annotated harness coverage now encode that accepted multi-expression subscript surface
- Active slice: prove chained projection after multi-expression subscripts for Cursor-shaped surfaces such as `cursor[i, j].value`

---

## [~] Track: Parser Regression Test Pass - Fix EBNF & Emitter for Full Cpp2 Support
*Link: [./conductor/tracks/parser_regression_pass_20260110/](./conductor/tracks/parser_regression_pass_20260110/)*
*Status: IN PROGRESS*

**Objective:** Fix cppfort parser and emitter to pass all cppfront regression tests with complete Cpp2 EBNF grammar support

**Scope:**
- Fix unified declarations (`name: type = init`)
- Fix parameter qualifiers (`inout`, `out`, `move`, `forward`)
- Fix function declarations and bodies
- Fix Pratt expression parser
- Fix all statement types
- Remove C++1 passthrough bypass
- Create tests for Advanced Cpp2 features (contracts, pattern matching, metafunctions, string interpolation, UFCS, templates, type system)
- Back-annotate Clang AST semantics into parse graph
- Use AST loss from corpus as validation metric

**Acceptance Criteria:**
- All 159 cppfront regression tests pass
- Generated C++ is functionally equivalent to cppfront output
- Performance: completes in under 5 minutes

---
