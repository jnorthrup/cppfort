# Cppfort Architecture: Stage0, Stage1, Stage2 with Inductive Graph Nodes

## Overview

Cppfort implements a self-hosting Cpp2 compiler with three stages and anticheat attestation through inductive regression testing. The system transpiles Cpp2 source to canonical C++, validates semantic equivalence through side-by-side execution, and extracts graph-node signals for model priming.

## Stage0: C++ Emitter

**Purpose**: Emit canonical C++ from internal AST representation.

**Current State**:
- AST definitions in `src/stage0/ast.h` (FunctionDecl, TypeDecl, etc.)
- Partial emitter in `src/stage0/emitter.cpp` (missing complete implementations)
- Token definitions in `src/stage0/token.h`
- CMake build in `src/stage0/CMakeLists.txt`

**Issues Identified**:
- Emitter incomplete: missing implementations for `emit_function`, `emit_block`, `emit_statement`, `emit_type`
- Generated C++ has syntax errors (e.g., `unique.new<int>(1)` instead of `std::make_unique<int>(1)`)
- No canonical formatting guarantees
- Missing unit tests

**Improvements Needed**:
- Complete emitter implementations with proper C++ syntax
- Canonical formatting (consistent indentation, spacing)
- Comprehensive unit tests
- Error handling for invalid AST nodes

## Stage1: Cpp2→C++ Transpiler

**Purpose**: Parse Cpp2 syntax and transpile to Stage0 AST for emission.

**Architecture**:
```
Cpp2 Source → Lexer → Parser → Stage0 AST → Stage0 Emitter → Canonical C++
```

**Components Needed**:
- **Lexer**: Tokenize Cpp2 source (extend `src/stage0/token.h` with Cpp2 keywords)
- **Parser**: Parse tokens into Stage0 AST (recursive descent or similar)
- **AST Mapping**: Transform Cpp2 constructs to C++ equivalents
- **Integration**: Use Stage0 emitter for final C++ output

**Key Mappings**:
- `main: () -> int = { ... }` → `int main() { ... }`
- `unique.new<T>(args)` → `std::make_unique<T>(args)`
- `shared.new<T>(args)` → `std::make_shared<T>(args)`
- `std::expected<T,E>` → proper expected implementation
- Function expressions → lambdas or function objects

**Testing**: Integration with regression harness for side-by-side validation.

## Stage2: Anticheat Disassembly & Attestation

**Purpose**: Extract binary features for anticheat attestation and reverse-engineering validation.

**Architecture**:
```
Compiled Binaries → Disassembler → Feature Extractor → Attestation Engine → Graph Nodes
```

**Components**:
- **Disassembler**: Use `objdump` or `llvm-objdump` to extract assembly
- **Feature Extractor**: Analyze instruction patterns, control flow, data sections
- **Attestation Module**: Generate cryptographic attestations of binary properties
- **Graph Node Generator**: Extract signals for inductive model

**Features Extracted**:
- Instruction counts by type (arithmetic, control flow, memory ops)
- Basic block complexity metrics
- Function call graphs
- Data section analysis
- Symbol table features

**Integration**: Runs after successful compilation in regression harness.

## Graph-Node Model for Inductive Learning

Graph nodes capture equivalence signals and metrics for training the inductive model that validates compilation correctness and detects anomalies.

### Node Structure

```json
{
  "test_file": "example.cpp2",
  "signals": {
    "transpile_equiv": true,    // Both stages transpile successfully
    "compile_equiv": true,     // Both produce compilable C++
    "run_equiv": true,         // Both binaries run without errors
    "output_equiv": true,      // Both produce identical output
    "semantic_equiv": true     // All above are true
  },
  "metrics": {
    "stage0_binary_size": 24576,
    "stage1_binary_size": 24576,
    "binary_size_ratio": 1.0,
    "stage0_disasm_lines": 1247,
    "stage1_disasm_lines": 1247,
    "disasm_ratio": 1.0
  },
  "features": {
    "cpp_diff_type": "identical|differ|null",
    "output_diff_type": "identical|differ|null",
    "stage0_success": true,
    "stage1_success": true,
    "error_patterns": ["unique.new", "expected<T,E>"],
    "anticheat_features": {
      "instruction_count": 1247,
      "basic_blocks": 89,
      "function_calls": 23
    }
  }
}
```

### Signal Definitions

- **transpile_equiv**: Both stages successfully transpile to C++
- **compile_equiv**: Both transpiled outputs compile successfully
- **run_equiv**: Both compiled binaries execute without runtime errors
- **output_equiv**: Both binaries produce identical stdout/stderr
- **semantic_equiv**: All equivalence signals are true (perfect semantic preservation)

### Metric Calculations

- **binary_size_ratio**: stage1_size / stage0_size (should be ~1.0 for equivalent semantics)
- **disasm_ratio**: stage1_disasm_lines / stage0_disasm_lines (should be ~1.0 for equivalent codegen)

### Feature Extraction

- **cpp_diff_type**: Result of diffing transpiled C++ outputs
- **output_diff_type**: Result of diffing program outputs
- **error_patterns**: Common error patterns from failed compilations
- **anticheat_features**: Binary analysis results from Stage2

## Data Flow

```
Cpp2 Test Corpus
    ↓
Stage0 Transpile → Stage1 Transpile
    ↓              ↓
Compile Both    Compile Both
    ↓              ↓
Run Both       Run Both
    ↓              ↓
Compare Outputs  Compare Outputs
    ↓              ↓
Extract Features  Extract Features
    ↓              ↓
Generate Graph Nodes
    ↓
Prime Inductive Model
```

## Development Workflow

1. **Stage0 Completion**: Finish emitter, add tests, ensure canonical C++ output
2. **Stage1 Implementation**: Build parser, implement AST mappings, integrate with Stage0
3. **Regression Validation**: Run harness, collect baseline graph nodes
4. **Stage2 Integration**: Add disassembly, feature extraction, attestation
5. **Model Priming**: Generate comprehensive dataset for inductive learning
6. **CI/CD**: Automate regression runs, model updates, and validation

## Risk Mitigation

- **Stage0 Issues**: Current emitter generates invalid C++ - fix before Stage1
- **Stage1 Complexity**: Cpp2→C++ mapping is non-trivial - incremental implementation
- **Stage2 Dependencies**: Binary analysis tools may not be available - graceful degradation
- **Model Priming**: Start with simple signals, expand feature set iteratively

## Success Criteria

- Stage0 emits valid, canonical C++ for all test cases
- Stage1 produces semantically equivalent C++ to Stage0
- Regression harness runs reliably with comprehensive coverage
- Graph nodes capture meaningful equivalence signals
- Anticheat features provide useful binary analysis
- Inductive model can detect compilation anomalies