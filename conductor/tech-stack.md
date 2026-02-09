# Technology Stack

## Core Language

- **C++23/C++2c** (Current): Clang 21.1.8 with `-std=c++2c` (`__cplusplus = 202400`)
- **C++26** (Target): Migration path for reflection, contracts, pattern matching when compiler support arrives
- **MLIR**: Multi-Level Intermediate Representation framework

### C++ Standard Strategy

- **Implementation**: C++23 features (ranges, concepts, coroutines, format, expected)
- **Forward Compatibility**: Design for C++26, implement with C++23 fallbacks
- **Migration Timeline**: Phase 9 (C++26 Integration) uses conditional compilation until features stabilize

## Build System

- **CMake 3.28+**: Build configuration with Homebrew LLVM integration
- **Ninja**: Fast parallel builds (recommended)
- **CTest**: Test execution framework

### CMake Targets

| Target | Type | Description |
|--------|------|-------------|
| `cppfort` | Executable | Main Cpp2 transpiler with MLIR pipeline |
| `Cpp2Transpiler` | Static Library | Transpiler core library |
| `cppfront` | Executable | Reference Cpp2 transpiler (from cppfront submodule) |
| `corpus_transpile` | Custom Target | Transpile 189 .cpp2 → .cpp files |
| `corpus_ast` | Custom Target | Generate Clang AST dumps from .cpp files |
| `corpus_reference` | Custom Target | Combined corpus processing (transpile + AST) |
| `test` | CTest | Run all test suites |

### Build Configuration

- **Homebrew LLVM**: `/opt/homebrew/opt/llvm` on macOS
- **libc++**: Homebrew's libc++ (not system SDK)
- **TableGen**: MLIR dialect code generation to `build/`
- **Output Directory**: `build/` (all artifacts, no generated files in root)

## Compilers & Toolchains

- **Clang 21.1.8** (LLVM/Homebrew): Primary C++ compiler, AST analysis
- **g++-15**: Alternative compiler (used for cppfront reference build)
- **cppfront** (hsutter/cppfront): Reference Cpp2 transpiler for semantic validation

## MLIR Infrastructure

- **LLVM 21.1.8**: Core LLVM libraries
- **MLIR Dialects**:
  - `cpp2fir` (Cpp2FIR): Front-IR for straight-line expression evaluation
  - `cpp2son` (Cpp2SON): Sea-of-Nodes IR with SSA and control flow
  - `mlir::func`: Function dialect integration
- **TableGen**: MLIR operation definition generator
- **mlir-opt**: MLIR optimization pipeline runner

## Testing & Analysis

- **CTest**: Test framework (21+ test suites verified passing)
  - `cpp26_contracts_test`: C++26 contract attribute parsing (4 tests)
  - `arena_inference_test`: Arena allocation inference (6 tests)
  - `allocation_strategy_test`: Allocation strategy verification (11 tests)
- **Integration Tests**: `test_cmake_integration.cpp` (10 build system tests)
- **lcov**: Code coverage measurement
- **genhtml**: Coverage report generation
- **LLVM gcov**: Coverage data collection
- **timeout**: Test execution time limit enforcement

## Regression Testing Infrastructure

- **Corpus**: 189 .cpp2 files from cppfront regression suite
  - 139 pure2 files (pure Cpp2 syntax)
  - 50 mixed files (Cpp2 + C++1 syntax)
- **Test Framework**: `cppfront_full_regression`, `regression_runner`
- **Hash Verification**: SHA-256 database for corpus integrity
- **AST Analysis**: Clang AST dumps (10GB reference data)
- **Isomorph Extraction**: 1.4M patterns extracted, 13.5K unique
- **MLIR Tagging**: 100% coverage with region type mapping
- **Semantic Loss Scoring**: Graph edit distance, type distance, operation distance
  - **Average Loss**: 0.124 (vs Target < 0.15)
  - **Pass Rate**: 98.4% pure2, 100% mixed

## Analysis Tools (Python 3)

- `tools/extract_ast_isomorphs.py`: Clang AST → graph pattern extraction
- `tools/tag_mlir_regions.py`: AST pattern → MLIR region mapping
- `tools/build_isomorph_database.py`: Deduplicated pattern database builder
- `tools/score_semantic_loss.py`: Reference vs candidate comparison
- `tools/aggregate_corpus_scores.py`: Corpus-wide metric aggregation
- `tools/generate_ast_mappings.py`: Batch AST dump generation
- `tools/process_corpus_with_cppfront.sh`: Corpus transpilation automation
- `tools/score_corpus_semantics.sh`: Automated scoring pipeline

## Version Control & Workflow

- **Git**: Source control
- **git notes**: Verification report attachment for checkpoints
- **Conductor**: Context-Driven Development framework
  - Spec → Plan → Implement → Checkpoint workflow
  - Test-Driven Development enforcement
  - Phase-based verification protocol

## Development Libraries

### Currently Available (C++23/C++2c - Clang 21.1.8)

- ✅ **Ranges**: `std::ranges` (202406) - algorithms, views, pipelines
- ✅ **Concepts**: `concept`, `requires` (202002) - type constraints
- ✅ **Format**: `std::format` (202110) - type-safe string formatting
- ✅ **Coroutines**: `co_await`, `co_yield`, `co_return` (201902) - async primitives
- ✅ **Expected**: `std::expected<T, E>` (202211) - result type with error handling
- ✅ **Optional/Variant**: `std::optional`, `std::variant` - sum types
- ✅ **Span**: `std::span` - safe array views
- ✅ **String_view**: `std::string_view` - zero-copy string operations
- ✅ **Reflection-driven SBO sizing**: `cpp2::reflection_sbo_size<T>()` with template metaprogramming fallback
- ✅ **Contracts parsing**: C++26 `[[expects]]` attribute parsing with AST integration
- ✅ **Pattern matching state tracking**: `ResourceState` enum with exhaustive matching

### Implemented (C++23 Fallbacks)

- ✅ **Reflection SBO**: Template metaprogramming fallback (`cpp2::reflection_sbo.hpp`)
  - **Migration**: Replace with `std::meta` when `__cpp_static_reflection` is available
- ✅ **Contracts**: AST-based parsing (`cpp26_contracts_test.cpp`)
  - **Migration**: Switch to native `[[expects]]` attribute parsing when available
- ✅ **Pattern Matching**: C++23 switch-based exhaustive matching (`ResourceState` enum)
  - **Migration**: Replace with `inspect` expressions when `__cpp_pattern_matching` is available
- ✅ **std::inplace_vector**: Custom `cpp2::inplace_vector<T, N>` implementation
  - **SBO sizing**: Automatic capacity via `sbo_capacity<T>()`
  - **Tests**: `reflection_sbo_test.cpp`, `inplace_vector_codegen_test.cpp`

### Planned (C++26 - Not Yet Available)

- ⏳ **std::execution**: Structured concurrency (senders/receivers)
  - **Fallback**: `std::coroutine` (available) for coroutine frame elision
  - **Use Case**: Port Kotlin `CoroutineScope` semantics

## Platform

- **macOS 14.6** (Darwin 24.6.0): Development environment
- **Homebrew**: Package management for LLVM/Clang

## Documentation Generation

- **Markdown**: All documentation in GitHub-flavored Markdown
- **Mermaid** (planned): Architecture diagrams
- **Graphviz** (via d3-graphviz.js): AST/IR visualization in docs

## Semantic Analysis Stack (Planned)

- **Escape Analysis Engine**: Lifetime and escape point tracking
- **Borrow Checker**: Rust-like ownership enforcement
- **Lifetime Region Analysis**: Scope-based lifetime bounds
- **Memory Transfer Tracker**: GPU/DMA transfer validation
- **Channel Safety Validator**: Concurrency data race detection

## JIT Memory-Managed Front-IR Architecture (Phases 7-10)

### Feature Validation (2026-01-06)

**Environment**: Clang 21.1.8, LLVM 21.1.8, macOS 14.6 (arm64)

#### ✅ Available Now (C++23/C++2c)

- Ranges (202406), Concepts (202002), Format (202110)
- Coroutines (201902), Expected (202211)
- Standard Library: optional, variant, span, string_view

#### ⏳ Planned (C++26 - Not Yet Available)

- Reflection (`std::meta`) - Feature test macro: `__cpp_static_reflection`
- Contracts (`[[expects]]`, `[[ensures]]`) - Feature test macro: `__cpp_contracts`
- Pattern Matching (`inspect`) - Feature test macro: `__cpp_pattern_matching`
- `std::execution` senders/receivers - Feature test macro: `__cpp_lib_execution`

### Phased Implementation Strategy

**Phase 7-8: Immediate (C++23 Compatible)**

- ✅ Arena allocation (`ArenaInferencePass`, `ArenaRegion`)
- ✅ Coroutine frame elision (leverages `std::coroutine` - available)
- ✅ MLIR passes (`CoroutineFrameSROA`) - no C++26 dependencies

**Phase 9: Conditional (C++26 Features with Fallbacks)**

1. **Reflection (`std::meta`)**:
   - **Fallback**: Template metaprogramming for SBO sizing
   - **Implementation**: `#ifdef __cpp_static_reflection` guards
   - **Migration**: Replace templates with `std::meta` when available

2. **Contracts**:
   - **Fallback**: Parse custom `// [[expects]]` comments in AST
   - **Implementation**: AST comment extraction → `ContractInfo` metadata
   - **Migration**: Switch to native `[[expects]]` attribute parsing

3. **Pattern Matching**:
   - **Fallback**: C++23 `std::expected` + exhaustive `if-else` chains
   - **Implementation**: Manual state machine for `ResourceState` tracking
   - **Migration**: Replace with `inspect` expressions

4. **std::execution**:
   - **Fallback**: Use `std::coroutine` directly (already available)
   - **Implementation**: Manual sender/receiver pattern implementation
   - **Migration**: Adopt `std::execution::schedule` when standardized

**Phase 10: Codegen (C++23 Compatible)**

- ✅ `AllocationStrategyPass` (no C++26 dependencies)
- ✅ Arena boilerplate generation (`cpp2::monotonic_arena<scopeID>`)
- ✅ Stack/arena/heap decision logic (standard C++23)

### Expected Timeline

- **Clang 22+**: Possible experimental reflection support
- **Clang 23+**: Possible contracts/pattern matching
- **Migration**: Incremental replacement of fallbacks with native features

## External Dependencies

- **third_party/cppfront**: Reference transpiler (submodule)
  - Source: `source/cppfront` binary (2.6MB, built with g++-15)
  - Regression tests: `regression-tests/` (189 .cpp2 files)
  - Reference corpus: `corpus/reference/` (158 transpiled outputs)
  - Reference ASTs: `corpus/reference_ast/` (10GB Clang dumps)

## Performance & Optimization

- **SCCP Pass**: Sparse Conditional Constant Propagation
  - Code coverage: 72.9% (250/343 lines), 100% functions (42/42)
  - Lattice-based dataflow analysis
  - Dead code elimination
  - Constant folding (arithmetic, logical, comparison operations)
- **Debug Logging**: LLVM_DEBUG infrastructure (DEBUG_TYPE="fir-sccp")
- **Coverage Instrumentation**: --coverage flag support

## Project Structure

```
cppfort/
├── conductor/              # Context-Driven Development framework
│   ├── product.md         # Vision, objectives, metrics
│   ├── tech-stack.md      # This file
│   ├── workflow.md        # Development process
│   ├── tracks.md          # Active feature tracks
│   └── tracks/            # Track specs and plans
├── docs/                   # Technical documentation
├── include/                # Header files (.h, .hpp)
├── src/                    # Implementation (.cpp)
├── tests/                  # Test suite
├── corpus/                 # Regression test corpus
│   ├── inputs/            # 189 .cpp2 source files
│   ├── reference/         # cppfront C++ outputs
│   ├── reference_ast/     # Clang AST dumps
│   ├── isomorphs/         # Extracted patterns (regenerable)
│   ├── tagged/            # MLIR-tagged patterns (regenerable)
│   └── database/          # Deduplicated database (regenerable)
├── tools/                  # Analysis and automation scripts
├── third_party/            # External dependencies
│   └── cppfront/          # Reference transpiler
├── build/                  # CMake build directory
└── mlir/                   # MLIR dialect definitions
```

## Build Configuration Flags

- `CMAKE_BUILD_TYPE`: Debug (with coverage) or Release
- `ENABLE_COVERAGE`: ON/OFF (code coverage instrumentation)
- `CMAKE_CXX_STANDARD`: 23 (current), 26 (target when compiler support available)
  - **Compile Flag**: `-std=c++2c` (enables C++23 + draft C++26 features)
  - **Validation**: `__cplusplus = 202400` (C++2c draft)
- `CMAKE_CXX_COMPILER`: clang++ or g++
- Coverage flags: `--coverage` (gcov/lcov compatible)

## Key Metrics

- Total lines of code: ~18,000 (excluding generated)
- Test coverage: 72.9% (SCCP), target >80%
- Build time: ~2 minutes (full rebuild with Ninja -j8)
- Corpus size: 189 files, 10GB AST data, 1.4M isomorphs
- Unique patterns: 13,545 (deduplication ratio 1.0%)
- Average Semantic Loss: 0.124 (Goal < 0.15)
- Regression Pass Rate: 98.4% (pure2), 100% (mixed)
