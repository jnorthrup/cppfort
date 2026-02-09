# Product Guide

## Vision

Comprehensive Cpp2-to-C++ transpiler with MLIR Front-IR, Sea-of-Nodes backend, and semantic preservation via Clang AST analysis.

## Core Objectives

1. **Full Cpp2 Language Support**: All major features (UFCS, contracts, metafunctions, pattern matching)
2. **Semantic Isomorphism**: Preserve Cpp2 semantics in generated C++ via Clang AST diffusion from cppfront reference
3. **MLIR Front-IR Pipeline**: Cpp2 → FIR dialect → SON dialect → optimizations → C++
4. **Safety-First**: Escape analysis, borrowing, bounds checking, contract validation
5. **Performance**: SCCP optimization, Sea-of-Nodes IR, lifecycle-based memory management

## Architecture (5 Layers)

```
Cpp2 Source
    ↓
[Lexer/Parser] → Cpp2 AST (with SemanticInfo)
    ↓
[Semantic Analysis] → Escape/Borrow/Ownership tracking
    ↓
[MLIR FIR Dialect] → Front-IR (expression-level)
    ↓
[MLIR SON Dialect] → Sea-of-Nodes IR (SSA, control flow)
    ↓
[SCCP + Optimizations] → Dead code elimination, constant propagation
    ↓
[Code Generator] → C++20/23 output
```

## Current State

### ✅ Working

- **Lexer/Parser**: Full Cpp2 grammar (pure2 mode)
  - Grammar: `conductor/PARSER_ORCHESTRATION.md` (EBNF → combinator mappings, single source of truth)
  - EBNF: `grammar/cpp2.ebnf` (formal grammar)
- **AST**: Complete node definitions with semantic info scaffolding
- **Type System**: Deduction, templates, UFCS resolution
- **MLIR Dialects**: FIR (Front-IR) + SON (Sea-of-Nodes) operational
- **SCCP Pass**: Sparse Conditional Constant Propagation (72.9% code coverage)
- **Code Generation**: pure2 files transpile successfully
- **Safety Checks**: Bounds, null, division-by-zero, integer overflow warnings
- **Metafunctions**: 14+ implemented (@value, @ordered, @interface, @regex, @autodiff, etc.)
- **Test Infrastructure**: 17 unit tests passing, regression framework operational
- **Reference Corpus**: Lazy-cached cppfront output + Clang AST dumps in `tests/reference/`
- **Semantic Analysis**: Escape analysis, borrowing, ownership, channels, external memory
- **Corpus Validation**: 178/189 (93.7%) passing, 0.124 avg semantic loss

### 🔧 In Progress

- **Code Quality**: Reducing output noise, improving formatting

### 📊 Corpus Analysis

- **Reference Corpus**: 158 transpiled tests in `tests/reference/` (lazy-cached via `tools/reference_corpus.sh`)
- **Head-to-Head**: 178/189 (93.7%) passing against cppfront (`VALIDATION_REPORT.md`, `SEMANTIC_PRESERVATION_REPORT.md`)
- **AST Database**: 1.4M isomorphs extracted, 13.5K unique patterns, 100% MLIR region coverage
- **Semantic Loss**: 0.124 (Avg), 0 High Loss Files (>0.15)
- **Failure Categories**: 2 known feature gaps (`pure2-last-use`, `pure2-print`)

## Guiding Principles

1. **Semantic Intent over Syntax**: Preserve Cpp2's safety and ownership semantics in C++
2. **Developer Experience**: TypeScript-like ergonomics, not traditional C++ idioms
3. **Isomorphic Mapping**: Clang AST from cppfront output guides Cpp2 semantic assignments
4. **Safety by Default**: Escape analysis, borrow checking, bounds validation automatic
5. **Performance Through IR**: Sea-of-Nodes enables aggressive optimization

## Key Features

### Language Support

- Unified declaration syntax (`name: type = value`)
- Parameter qualifiers (`in`, `out`, `inout`, `move`, `forward`)
- UFCS (Unified Function Call Syntax)
- Contracts (preconditions, postconditions, assertions)
- Metafunctions (compile-time code generation)
- Template support
- String interpolation

### Safety Features (Implemented)

- Null pointer checking
- Array bounds checking
- Division by zero prevention
- Mixed-sign arithmetic warnings
- Use-after-move detection
- Integer overflow warnings (Safety 6)
- Escape Analysis (Heap/Return/Channel/GPU/DMA)
- Borrow Checking (Exclusive mutable borrows)
- Channel Safety (Ownership transfer, data race detection)
- External Memory Safety (Async DMA aliasing checks)
- Lifetime Regions (Borrow outlives owner enforcement)
- Scope-Inferred Arenas (JIT memory management via ArenaInferencePass)
- Coroutine Frame Elision (Stack allocation for non-escaping frames)
- C++26 Integration (Reflection SBO sizing, Contracts, Pattern matching)

### Advanced Features

- **CAS-Driven Modules**: Markdown block comments trigger C++20 module generation
- **Fractal JIT**: Content-addressable caching for JIT-compiled code
- **Concurrency**: Kotlin-style channels, coroutine scopes, spawn/await
- **GPU Kernels**: Parallel loops with launch config, memory policy annotations
- **External Memory Pipeline**: DMA buffers, memory regions (CPU/GPU global/shared/constant)

## Target Users

C++ developers seeking:

- Modern syntax with safety guarantees
- Migration from C++ to Cpp2
- Performance-critical systems with memory safety
- Concurrent/parallel programming with channels
- GPU kernel development

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| pure2 tests passing | 127/129 (98.4%) | >95% |
| mixed tests passing | 50/50 (100%) | 50/50 |
| Parameter semantics correct | 100% | 100% |
| Average corpus semantic loss | 0.124 | <0.15 |
| SCCP code coverage | 72.9% | >80% |
| Escape analysis coverage | 100% | 100% |

## Documentation

- **conductor/PARSER_ORCHESTRATION.md**: EBNF design → combinator mappings → AST isomorphs → loss scoring (single source of truth)
- **conductor/tech-stack.md**: Tools and infrastructure
- **conductor/workflow.md**: Development process and quality gates
- **conductor/tracks.md**: Active feature tracks
- **docs/cpp2/**: Cpp2 language reference (external)
- **docs/cppfront/**: Cppfront reference (external)
- **docs/sea-of-nodes/**: Sea-of-Nodes reference (external)
- **docs/Simple/**: Simple IR reference (external)
