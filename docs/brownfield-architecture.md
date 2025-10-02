# Cppfort Brownfield Architecture Document

## Introduction

This document captures the CURRENT STATE of the Cppfort codebase, including technical debt, workarounds, and real-world patterns. Cppfort is a self-hosting Cpp2 compiler with three stages and anticheat attestation through inductive regression testing. The system transpiles Cpp2 source to canonical C++, validates semantic equivalence through side-by-side execution, and extracts graph-node signals for model priming.

### Document Scope

Comprehensive documentation of the entire Cppfort system, focusing on the three-stage pipeline: Stage0 (C++ infrastructure), Stage1 (Cpp2→C++ transpiler), and Stage2 (anticheat attestation).

### Change Log

| Date   | Version | Description                 | Author    |
| ------ | ------- | --------------------------- | --------- |
| 2025-10-02 | 1.0     | Initial brownfield analysis | BMAD Orchestrator |

## Quick Reference - Key Files and Entry Points

### Critical Files for Understanding the System

- **Main Entry**: `src/nway_compiler.cpp` (main compiler entry point)
- **Configuration**: `CMakeLists.txt` (build configuration)
- **Core Business Logic**: `src/stage0/` (AST, emitter), `src/stage1/` (transpiler), `src/stage2/` (attestation)
- **API Definitions**: `include/cpp2.h`, `include/cpp2_impl.h`
- **Key Algorithms**: `src/stage0/emitter.cpp` (C++ emission), `src/stage2/anticheat_cli` (disassembly)
- **Build System**: `CMakeLists.txt`, `build/` directory

### Key CLIs and Commands

- `stage0_cli` – emits C++ from a `.cpp2` file
- `stage1_cli` – transpiles a `.cpp2` file to C++ (wrapper around Stage 0)
- `anticheat` – command-line helper executing the attestation

## High Level Architecture

### Technical Summary

Cppfort implements a three-stage pipeline for the cpp2 language:

- **Stage 0**: Core C++ infrastructure (AST, Emitter, documentation support)
- **Stage 1**: cpp2 → C++ transpiler using Stage 0 emitter
- **Stage 2**: Anti-cheat attestation through binary disassembly and hashing

The system uses Sea of Nodes IR concepts following the Simple compiler tutorial approach, with meta-programming transformations for multiple target emissions.

### Actual Tech Stack (from CMakeLists.txt and source)

| Category  | Technology | Version | Notes                      |
| --------- | ---------- | ------- | -------------------------- |
| Language  | C++20      | C++20   | Required standard         |
| Compiler  | Clang      | Latest  | Forced libc++ for ABI      |
| Build     | CMake      | 3.16+   | Out-of-source builds       |
| IR        | LLVM/MLIR  | Dev     | Homebrew integration       |
| Testing   | Custom     | N/A     | Regression harness         |

## Source Tree and Module Organization

### Project Structure (Actual)

```text
cppfort/
├── src/
│   ├── nway_compiler.cpp     # Main compiler entry
│   ├── stage0/               # C++ infrastructure
│   │   ├── ast.h            # AST definitions
│   │   ├── emitter.cpp      # C++ emitter (incomplete)
│   │   ├── token.h          # Token definitions
│   │   └── CMakeLists.txt   # Stage0 build
│   ├── stage1/               # Transpiler
│   │   └── main.cpp2        # Sample cpp2 file
│   ├── stage2/               # Anticheat
│   │   └── anticheat_cli    # Attestation tool
│   ├── cpp2/                 # Cpp2 language features
│   ├── ir/                   # Intermediate representations
│   ├── mlir_bridge/          # MLIR integration
│   ├── parsers/              # Parser implementations
│   ├── utils/                # Utility functions
│   ├── compat/               # Compatibility layer
│   └── attestation/          # Attestation modules
├── include/
│   ├── cpp2.h               # Public API
│   └── cpp2_impl.h          # Implementation headers
├── regression-tests/         # Test harness
├── docs/                     # Documentation
└── build/                    # Build artifacts
```

### Key Modules and Their Purpose

- **Stage0 (`src/stage0/`)**: Core C++ infrastructure with AST definitions and emitter. Currently incomplete - missing full emitter implementations.
- **Stage1 (`src/stage1/`)**: Cpp2→C++ transpiler. Currently minimal - builds basic TranslationUnit and uses Stage0 emitter.
- **Stage2 (`src/stage2/`)**: Anticheat attestation. Disassembles binaries and generates SHA-256 hashes for verification.
- **NWay Compiler (`src/nway_compiler.cpp`)**: Main entry point coordinating the three stages.
- **MLIR Bridge (`src/mlir_bridge/`)**: Integration with LLVM/MLIR for advanced IR transformations.
- **Regression Tests (`regression-tests/`)**: Triple induction testing framework validating all stages.

## Data Models and APIs

### Data Models

AST nodes defined in `src/stage0/ast.h`:

- FunctionDecl, TypeDecl, VarDecl
- Statement types (BlockStmt, ExprStmt, etc.)
- Expression types (BinaryExpr, CallExpr, etc.)

### API Specifications

- **Public API**: `include/cpp2.h` - main interface for cpp2 compilation
- **Implementation**: `include/cpp2_impl.h` - internal implementation details
- **CLI Interfaces**: stage0_cli, stage1_cli, anticheat command-line tools

## Technical Debt and Known Issues

### Critical Technical Debt

1. **Incomplete Emitter (Stage0)**: `src/stage0/emitter.cpp` missing implementations for `emit_function`, `emit_block`, `emit_statement`, `emit_type`
2. **Syntax Errors in Generated C++**: Emitter produces invalid syntax like `unique.new<int>(1)` instead of `std::make_unique<int>(1)`
3. **Minimal Parser (Stage1)**: Current implementation builds only basic TranslationUnit, no full Cpp2 parsing
4. **No Unit Tests**: Missing comprehensive unit tests for core components
5. **Hardcoded Dependencies**: CMakeLists.txt forces libc++ and has hardcoded LLVM paths

### Workarounds and Gotchas

- **Compiler Requirements**: Must use Clang with libc++ due to ABI issues
- **Build Directory**: Must use out-of-source builds (`mkdir build && cd build && cmake ..`)
- **LLVM Integration**: Depends on Homebrew LLVM/MLIR installation paths
- **Testing**: Triple induction framework requires manual script execution

## Integration Points and External Dependencies

### External Services

| Service  | Purpose  | Integration Type | Key Files                      |
| -------- | -------- | ---------------- | ------------------------------ |
| LLVM/MLIR| IR transformations | Direct linking   | `src/mlir_bridge/`, CMakeLists.txt |
| objdump  | Binary disassembly | System calls     | `src/stage2/anticheat_cli`     |
| Homebrew | Package management | Build deps       | CMakeLists.txt                 |

### Internal Integration Points

- **Stage Pipeline**: Stage1 uses Stage0 emitter, Stage2 validates Stage1 output
- **Triple Induction**: Regression tests validate semantic equivalence across stages
- **MLIR Bridge**: Provides advanced IR capabilities for Sea of Nodes implementation

## Development and Deployment

### Local Development Setup

1. Install LLVM/MLIR via Homebrew
2. Use out-of-source CMake build:

   ```bash
   mkdir -p build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Debug
   cmake --build .
   ```

3. Run regression tests: `regression-tests/run_tests.sh`

### Build and Deployment Process

- **Build Command**: `cmake --build .` from build directory
- **Test Command**: `regression-tests/run_triple_induction.sh`
- **Deployment**: Manual - no automated deployment setup
- **Environments**: Local development only, no staging/production separation

## Testing Reality

### Current Test Coverage

- **Unit Tests**: None - missing comprehensive unit test suite
- **Integration Tests**: Triple induction framework via shell scripts
- **Regression Tests**: Basic comparison tests in `regression-tests/`
- **E2E Tests**: Manual verification of transpilation and attestation

### Running Tests

```bash
# Basic regression
regression-tests/run_tests.sh

# Triple induction (all stages)
regression-tests/run_triple_induction.sh

# Attestation tests
regression-tests/run_attestation_tests.sh
```

### Test Issues

- No automated test framework (Jest, Google Test, etc.)
- Tests are shell scripts with manual verification
- Missing unit tests for core AST and emitter logic
- No CI/CD integration for automated testing

## Appendix - Useful Commands and Scripts

### Frequently Used Commands

```bash
# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build .

# Transpile
./stage1_cli <input.cpp2> <output.cpp>

# Attest
./anticheat <binary_path>

# Test
regression-tests/run_tests.sh
regression-tests/run_triple_induction.sh
```

### Debugging and Troubleshooting

- **Build Issues**: Ensure LLVM/MLIR is installed via Homebrew
- **Emitter Errors**: Check generated C++ syntax manually
- **Attestation Failures**: Verify objdump is available and binary is compiled with debug symbols
- **Test Failures**: Compare Stage0 vs Stage1 output manually
