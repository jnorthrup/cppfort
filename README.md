# cppfort

**Cpp2 → C++ transpiler with MLIR-based Sea-of-Nodes IR pipeline**

## Overview

Cppfort is a transpiler for the Cpp2 language (from [cppfront](https://github.com/hsutter/cppfront)) that combines:
- **Traditional AST pipeline**: Isomorphic Cpp2 → C++ transpilation targeting semantic equivalence and `wdiff` compatibility with `cppfront` output.
- **SoN/MLIR pipeline**: Graph-based "borrow checker" implementing memory and lifecycle safety analysis using MLIR dialects (`sond`, `cpp2fir`).
- **Clang AST mapping**: Inverse inference from C++ AST to Cpp2 constructs to ground semantics.

## Project Objectives

1.  **Isomorphic C++ Transpilation**: Clean `wdiff` and whitespace compatibility with `cppfront` output.
2.  **Roundtrip Transpilation**: Strictly defined flow: **Cpp2 (or best C++) → MLIR → Cpp2/C++**. Uses MLIR as the central hub for optimization and safety analysis.
3.  **Safety Analysis (SON)**: Implementing memory/lifecycle safety checks (borrow checker) via Sea-of-Nodes IR.
4.  **Semantic Grounding**: Using Clang AST to reverse-map C++ semantic features to extend the Cpp2/Cppfort semantic foundation.

## Key Components

### MLIR Dialects
- **`cpp2fir` (Frontend IR)**: High-level IR for Cpp2 constructs.
- **`sond` (Sea-of-Nodes Dialect)**: Analysis IR for safety checks (memory, lifecycle, concurrency).
- **`cpp2` (Mapping Dialect)**: Used by `tools/inference/` for Clang AST mapping.

### Mapping Tools (`tools/inference/`)
Python toolchain for extracting Clang AST → MLIR op mappings:
- **emit_mappings.py**: Extract mapping candidates from C++ source
- **batch_emit_mappings.py**: Process multiple files, aggregate results
- **validate_against_dialect.py**: Validate mappings against dialect
- **run_inference.sh**: Wrapper with libclang configuration

### Documentation
- **[docs/OBJECTIVES.md](docs/OBJECTIVES.md)**: Detailed project goals and priorities.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Overall design.
- **[docs/MAPPING_SPEC.md](docs/MAPPING_SPEC.md)**: Mapping schema specification.
- **[docs/MAPPING_TASK.md](docs/MAPPING_TASK.md)**: Task definition.
- **[docs/MAPPING_PROGRESS.md](docs/MAPPING_PROGRESS.md)**: Implementation status.
- **[docs/sea-of-nodes/](docs/sea-of-nodes/)**: 24 chapters from Cliff Click's book.

## Quick Start

### Generate Mappings

```bash
# Single file
./tools/inference/run_inference.sh tools/inference/emit_mappings.py \
  -i tools/inference/samples/sample_son.cpp \
  -o mappings.json -- -std=c++20

# Batch process
python3 tools/inference/batch_emit_mappings.py \
  -i corpus/inputs \
  -o output_dir \
  --limit 10 \
  --aggregate
```

### Validate Mappings

```bash
python3 tools/inference/validate_against_dialect.py \
  -m mappings.json \
  -d include/Cpp2Dialect.td
```

## Building

### Prerequisites

```bash
# macOS (Homebrew LLVM)
brew install llvm ninja cmake

# Set LLVM in PATH (add to ~/.zshrc or ~/.bashrc)
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
```

### Build Commands

```bash
# Configure with CMake (Homebrew LLVM, Ninja)
cmake -B build -G Ninja .

# Build main targets
ninja -C build cppfort              # Main transpiler executable (slim, no MLIR deps)
ninja -C build Cpp2Transpiler       # Transpiler library (with MLIR support)
ninja -C build cppfront             # Cpp2 reference transpiler (from cppfront)

# Build and run tests
ninja -C build test                 # Run all CTest suites
./build/tests/cpp26_contracts_test  # Run specific test

# Corpus processing (requires cppfront)
ninja -C build corpus_transpile     # Transpile .cpp2 → .cpp
ninja -C build corpus_ast           # Generate AST dumps
ninja -C build corpus_reference     # Combined transpile + AST

# Clean build
rm -rf build && cmake -B build -G Ninja . && ninja -C build
```

### CMake Targets

| Target | Description |
|--------|-------------|
| `cppfort` | Main transpiler executable (slim parser, C++ output) |
| `Cpp2Transpiler` | Transpiler library (MLIR analysis backend) |
| `cppfront` | Cpp2 reference transpiler |
| `corpus_transpile` | Transpile 189 .cpp2 files |
| `corpus_ast` | Generate AST dumps |
| `corpus_reference` | Combined corpus processing |
| `test` | Run all test suites |

### Running the Transpiler

```bash
# Transpile a Cpp2 file to C++
./build/cppfort input.cpp2 -o output.cpp
```

## Project Status

### Completed
- ✅ Isomorphic transpiler (slim parser -> emitter)
- ✅ MLIR dialect definitions (`sond`, `cpp2fir`, `cpp2`)
- ✅ Mapping extraction toolchain
- ✅ Sample corpus and test infrastructure
- ✅ Documentation and usage guides

### In Progress
- 🔄 MLIR lowering (AST -> FIR -> SON) for safety checks
- 🔄 Full corpus validation (matching `cppfront` output)
- 🔄 Cpp2 file transpilation integration

### Planned
- 📋 Roundtrip validation (Cpp2 ↔ MLIR ↔ C++/Cpp2)
- 📋 CI/CD pipeline
- 📋 Sea-of-Nodes chapter pattern extraction

## Validation Results

Latest validation (sample_son.cpp, 6,301 mappings):
- **call**: 2,170 mappings (CallExpr)
- **func**: 1,497 mappings (FunctionDecl)
- **return**: 1,021 mappings (ReturnStmt)
- **var**: 677 mappings (VarDecl)
- **binop**: 599 mappings (BinaryOperator)
- **if**: 249 mappings (IfStmt)
- **while**: 46 mappings (WhileStmt)
- **for**: 42 mappings (ForStmt)

All mappings validate successfully against the dialect definition.

## Architecture

```
C++ Source (.cpp2) 
    ↓
[cppfront transpiler] → C++ (.cpp)
    ↓
[Clang AST parser]
    ↓
[Mapping Extractor] → Mapping Artifacts (JSON)
    ↓
[MLIR Emitter] → cpp2 Dialect Ops
    ↓
[SoN Optimizer] → Optimized MLIR
    ↓
[Safety Analysis] → Borrow Checker / Lifecycle Checks
```

## Dependencies

- **LLVM/MLIR**: 21.1+ (Homebrew LLVM on macOS)
- **Clang**: 21.1+ (from Homebrew LLVM)
- **CMake**: 3.28+ (for build system)
- **Ninja**: (recommended build tool)
- **Python**: 3.10+ with libclang bindings (for mapping tools)

## References

- [cppfront](https://github.com/hsutter/cppfront) - Herb Sutter's Cpp2 transpiler
- [Sea of Nodes IR](https://github.com/SeaOfNodes/Simple) - Cliff Click's SoN book/reference
- [MLIR](https://mlir.llvm.org/) - Multi-Level IR framework

## License

[License details to be determined]

## Contributing

See [.github/copilot-instructions.md](.github/copilot-instructions.md) for development guidelines.
