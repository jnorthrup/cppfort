# Spec: SCCP Pass Implementation for FIR Dialect

## 1. Overview

Implement a Sparse Conditional Constant Propagation (SCCP) pass for the Cpp2 FIR (Front-IR) dialect. SCCP is a dataflow analysis algorithm that propagates constant values through the IR and enables compile-time optimizations including constant folding and dead code elimination.

The SCCP pass will run on the FIR dialect before lowering to the SON (Sea of Nodes) dialect, enabling early optimization of the Cpp2 AST representation. This ensures borrowing and allocator magic remains upstream in the compilation pipeline.

## 2. Key Features & Requirements

### 2.1 Core SCCP Algorithm
- **Lattice-Based Analysis:** Implement a type lattice with TOP (unknown), BOTTOM (unreachable), and intermediate values (constants, ranges, types)
- **Worklist Algorithm:** Use a worklist-driven approach for efficient propagation
- **Forward Dataflow Analysis:** Propagate constants from definition to use sites
- **Conditional Propagation:** Only propagate values when control flow can reach the use site (sparse analysis)

### 2.2 Type Lattice Support
- **Basic Types:** Integer, Boolean, Control, Memory types
- **Range Analysis:** Track min/max ranges for integer values to enable more optimizations
- **Float Support:** Handle special float values (NaN, Infinity, -0.0)
- **Meet Operations:** Implement lattice meet at control flow merge points (phi nodes)
- **Type Unions:** Support union types for values from different control flow paths

### 2.3 Operation Coverage

#### Arithmetic Operations (Constant Folding)
- AddOp, SubOp, MulOp, DivOp - Fold when both operands are constants
- Unary MinusOp - Fold when operand is constant
- CmpOp - Fold to true/false when operands are constants
- Handle overflow and divide-by-zero cases

#### Logical Operations
- AndOp, OrOp - Fold with truth tables
- NotOp - Fold boolean negation

#### Control Flow Operations
- IfOp - Eliminate dead branches when condition is constant
- PhiOp - Merge constant values from live predecessors
- ReturnOp - Fold constant return values
- FuncOp - Analyze function calls with constant arguments (interprocedural)

### 2.4 MLIR Integration
- **Pass Registration:** Register as a standard MLIR pass using `PassRegistration`
- **Dialect Conversion:** Use MLIR's `RewritePattern` infrastructure
- **Analysis Manager:** Integrate with MLIR's `AnalysisManager` for pass caching
- **IR Mutation:** Use `PatternRewriter` for safe IR modifications

### 2.5 Pipeline Integration
- **Standalone Pass:** Can run independently via `mlir-opt --sccp`
- **Pipeline Integration:** Inserted after FIR generation, before FIR-to-SON lowering
- **Preserve Analyses:** Preserve dominance, post-dominance analyses for subsequent passes

## 3. Technical Approach

### 3.1 Core Data Structures

```cpp
// Lattice value for SCCP analysis
class LatticeValue {
  enum Kind { Top, Constant, Bottom, IntegerRange, FloatSpecial };
  Kind kind;
  Attribute value;  // MLIR Attribute for constant values
  // Range tracking for integers
  std::optional<int64_t> min;
  std::optional<int64_t> max;
};
```

### 3.2 Algorithm Outline
1. Initialize all values to TOP
2. Add all block arguments to worklist
3. Process worklist until empty:
   - For each operation, compute meet of operand lattice values
   - If value changed, add users to worklist
   - Fold constant operations
   - Eliminate dead branches

### 3.3 Constant Folding
- Use MLIR's `ConstOp` folder for built-in types
- Implement custom folders for FIR dialect operations
- Handle attributes: `IntegerAttr`, `FloatAttr`, `BoolAttr`, `UnitAttr`

## 4. Acceptance Criteria

### 4.1 Functional Requirements
- [ ] Constants propagate correctly through all arithmetic operations (Add, Sub, Mul, Div)
- [ ] Constants propagate through all logical operations (And, Or, Not)
- [ ] Comparisons with constant operands fold to boolean constants
- [ ] Constant conditions in `if` statements eliminate dead branches
- [ ] Phi nodes correctly merge constants from live predecessors
- [ ] Range analysis enables optimization of bounded integer operations

### 4.2 Testing Requirements
- [ ] Unit tests for each operation type (arithmetic, logical, comparison)
- [ ] Tests for control flow propagation (if/else, phi merging)
- [ ] Tests for range analysis and boundary conditions
- [ ] Integration tests with FIR test programs
- [ ] Tests for special cases (overflow, divide-by-zero, NaN, Infinity)

### 4.3 Integration Requirements
- [ ] Pass registers with MLIR pass pipeline
- [ ] Compatible with existing FIR-to-SON lowering pass
- [ ] No regressions in existing test suite
- [ ] Performance benchmarks show improvement or neutral impact

### 4.4 Code Quality
- [ ] Code coverage >20% for SCCP implementation
- [ ] Follows MLIR C++ coding style
- [ ] Documentation for public APIs
- [ ] Debug logging support for troubleshooting

## 5. Out of Scope

- Interprocedural SCCP across function boundaries (future enhancement)
- SCCP on SON dialect after lowering (separate pass)
- Memory alias analysis and load/store optimization
- Loop-specific optimizations (unrolling, vectorization)

## 6. Deliverables

1. **SCCP Pass Implementation:** `src/SCCP.cpp` with `createSCCPPass()` factory
2. **Lattice Analysis:** `include/LatticeValue.h` with lattice data structures
3. **Test Suite:** `tests/test_sccp_pass.cpp` with comprehensive tests
4. **Integration:** Pass registered in CMake and FIR dialect
5. **Documentation:** Comments and docstrings explaining the algorithm
