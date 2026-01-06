# Technology Stack

## Core Language
- **C++26**: Primary implementation language
- **MLIR**: Multi-Level Intermediate Representation framework

## Build System
- **CMake 3.25+**: Build configuration
- **Ninja**: Fast parallel builds
- **CTest**: Test execution framework

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
- **CTest**: Test framework (29 test suites, 24 passing)
- **lcov 2.4**: Code coverage measurement
- **genhtml**: Coverage report generation
- **LLVM gcov 17.0.0**: Coverage data collection
- **timeout (15s)**: Test execution time limit enforcement

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

## Analysis Tools (Python 3)
- `tools/extract_ast_isomorphs.py`: Clang AST → graph pattern extraction
- `tools/tag_mlir_regions.py`: AST pattern → MLIR region mapping
- `tools/build_isomorph_database.py`: Deduplicated pattern database builder
- `tools/score_semantic_loss.py`: Reference vs candidate comparison
- `tools/aggregate_corpus_scores.py`: Corpus-wide metric aggregation
- `tools/generate_ast_mappings.py`: Batch AST dump generation
- `tools/process_corpus_with_cppfront.sh`: Corpus transpilation automation

## Version Control & Workflow
- **Git**: Source control
- **git notes**: Verification report attachment for checkpoints
- **Conductor**: Context-Driven Development framework
  - Spec → Plan → Implement → Checkpoint workflow
  - Test-Driven Development enforcement
  - Phase-based verification protocol

## Development Libraries
- **Standard Library**: C++26 features (ranges, format, concepts, modules, reflection, contracts, pattern matching)
- **std::meta**: Compile-time reflection (`std::meta::info`, type introspection)
- **std::execution**: Structured concurrency (senders/receivers, `std::execution::schedule`)
- **Contracts**: `[[expects]]`, `[[ensures]]`, `[[assert]]` for precondition/postcondition enforcement
- **Pattern Matching**: `inspect` expressions for exhaustive matching
- **std::inplace_vector**: Stack-allocated vector with reflection-driven SBO sizing
- **Optional/Variant**: Sum type support
- **Span**: Safe array views
- **String_view**: Zero-copy string operations

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
- `CMAKE_CXX_STANDARD`: 26
- `CMAKE_CXX_COMPILER`: clang++ or g++
- Coverage flags: `--coverage` (gcov/lcov compatible)

## Key Metrics
- Total lines of code: ~18,000 (excluding generated)
- Test coverage: 72.9% (SCCP), target >80%
- Build time: ~2 minutes (full rebuild with Ninja -j8)
- Corpus size: 189 files, 10GB AST data, 1.4M isomorphs
- Unique patterns: 13,545 (deduplication ratio 1.0%)
