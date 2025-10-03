# Cppfort Architecture: Stage0, Stage1, Stage2 with Inductive Graph Nodes

## Overview

Cppfort implements a self-hosting Cpp2 compiler with three stages and anticheat attestation through inductive regression testing. The system transpiles Cpp2 source to canonical C++, validates semantic equivalence through side-by-side execution, and extracts graph-node signals for model priming.

### Implementation Strategy: Bands

The Sea of Nodes implementation is organized into **Bands** - coherent implementation phases that align with proven frontend IR concepts from LLVM, MLIR, and production compilers.

**Critical Divergence:** Cppfort diverges from Simple compiler's single-target approach. Cppfort uses Sea of Nodes as **source IR** for **n-way conversion** to multiple targets via TableGen pattern matching.

**Current Status:**

- ✅ **Band 1** (Chapters 1-6): Foundation - SSA form, basic nodes
- ✅ **Band 2** (Chapters 7-10): Loops + Memory - CFG, memory model
- ⚠️ **Band 3** (Chapter 11): Scheduling - GCM implementation (partial)
- ✅ **Band 4** (Chapters 12-15): Type System - Full type lattice
- ❌ **Band 5+** (Chapters 16-24): Advanced optimizations (not started)
- ❌ **N-Way Lowering** (Post-optimization): Pattern-based conversion to MLIR/C/C++/Rust/WASM

**See:**

- [Band Structure](architecture/band-structure.md) - Implementation milestones
- [Band-IR Alignment](architecture/band-ir-alignment.md) - Dovetailing with real-world IR concepts
- **[Divergence: Simple vs N-Way](architecture/divergence-son-simple-n-way.md)** - Why cppfort needs pattern matching

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

## Stage2: Decompilation & Differential Analysis Pipeline

**Purpose**: Extract assembly patterns across optimization levels, perform differential tracking to identify optimization transformations, and generate TableGen pattern databases for compiler optimization research.

**Architecture**:

```
CPP2 Source → Multi-Level Compilation (-O0/-O1/-O2/-O3) → Disassembly → ASM Parser → Differential Tracker → Pattern Database → TableGen Integration
```

**Components**:

- **Multi-Level Compiler**: Compile same source at different optimization levels
- **Disassembler**: Extract assembly using `objdump` or `llvm-objdump`
- **ASM Parser**: Parse assembly instructions into structured representation
- **Differential Tracker**: Compare instruction sets across optimization levels to identify transformations
- **Pattern Database**: Store optimization patterns and transformation rules
- **TableGen Exporter**: Generate TableGen definitions for pattern matching

**Features Extracted**:

- Assembly instruction patterns and sequences
- Optimization transformations (constant folding, dead code elimination, CSE)
- Control flow changes across optimization levels
- Memory access pattern optimizations
- Loop transformations and unrolling detection

**Integration**: Connected to regression harness for automated pattern extraction during testing.

**Status**: ⚠️ **Partially Implemented**

- **Phase 1 (Differential Analysis)**: ✅ **Implemented** - ASM parsing, Merkle differential tracking, optimization-level pattern survival analysis, build attestation and reproducibility verification
- **Phase 2 (Direct Decompilation)**: 🔄 **Redesigned** - Architecture-specific ASM→C++ lifting without Sea of Nodes IR complexity

**Current Capabilities**:

- Binary differential analysis across optimization levels
- Optimization pattern survival tracking
- Build reproducibility verification via Merkle roots
- Security attestation through SHA-256 + Merkle proofs

**Phase 2 Implementation (Corrected Architecture)**:

- **Phase 2A (x86-64)**: Architecture detector, x86-64 instruction analyzer, CFG recovery, variable inference, direct C++ generation
- **Phase 2B (ARM64)**: ARM64 instruction decoder, ARM-specific CFG recovery, ARM64 code generation
- **Success Target**: 60-90% decompilation accuracy for supported architectures

**Simplified Decompilation Architecture** (Revised Phase 2):

```text
Binary → Format Parser → Architecture Detector → ASM Analyzer → Control Flow Recovery → Data Flow Analysis → C++ Code Generator
```

**ARM/Intel-Focused Components Needed**:

- **Architecture Detector**: Identify ARM64, x86-64, ARM32, x86 instruction sets
- **ASM Analyzer**: Parse and categorize instructions by architecture (no IR lifting)
- **Control Flow Recovery**: Reconstruct if/while/for structures from conditional jumps
- **Data Flow Analysis**: Track register/memory usage, infer variable types
- **C++ Code Generator**: Direct assembly-to-C++ translation with architecture-specific patterns

See [Stage2 Disasm→TableGen Differential](architecture/stage2-disasm-tblgen-differential.md) for Phase 1 implementation details and [Idempotent Pattern Optimization](architecture/idempotent-pattern-optimization.md) for optimization strategies.

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

## Multi-Index Subsumption Engine

**Status:** Design Phase | **ADR:** [ADR-001](architecture/decisions/ADR-001-multi-index-subsumption-engine.md)

The subsumption engine provides unified query and projection capabilities for:

- **MLIR integration** - Pattern matching and dialect conversions
- **Sea of Nodes optimization** - Multi-criteria graph queries (type + CFG + data flow)
- **N-way projections** - Graph transformations via hierarchical subsumption rules
- **Node feature extraction** - Project node properties for analysis/codegen

### Core Requirements

1. **Hash-based primary index** - O(1) lookup by node ID
2. **Type hierarchy queries** - Subsumption lattice for type matching
3. **CFG topology queries** - Domination, loop depth, control flow
4. **Data flow projections** - Def-use chains, memory dependencies
5. **MLIR pattern matching** - TableGen-style rule application

### Integration Points

- **Band 3 GCM:** Loop-invariant code motion, dominator queries, anti-dependency detection
- **Band 4 Types:** Type lattice navigation, subtype queries, unification
- **Future MLIR:** Dialect lowering, operation legality, multi-level optimization

### API Preview

```cpp
// Loop-invariant code motion query
auto hoistCandidates = engine.query()
    .whereLoopDepth(lessThan(currentLoop->depth()))
    .whereDataFlow(allInputsAvailable(currentLoop->preheader()))
    .projectToSchedule();

// MLIR pattern matching
Pattern addPattern = engine.createPattern()
    .match<AddNode>()
    .whereType(isIntegral())
    .rewrite([](AddNode* n) {
        return mlir::arith::AddIOp(n->lhs(), n->rhs());
    });
```

**See:**

- [Subsumption Engine Architecture](architecture/subsumption-engine.md) for full details
- [Subsumption-Based Densification Gains](architecture/subsumption-densification-gains.md) for optimization analysis
- [Borrow Checking & Arena Allocators](architecture/borrow-checking-arenas.md) for memory safety and one-way drone allocation
- **[Subsumption Boundaries vs Bands](architecture/subsumption-boundaries-vs-bands.md)** - Critical: Band implementation phases ≠ Subsumption query partitions
- **[TableGen Optimization Rainbow Table](architecture/tblgen-optimization-rainbow-table.md)** - Optimization-level-aware lowering via differential equations
