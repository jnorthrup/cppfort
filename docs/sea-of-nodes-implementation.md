# Sea of Nodes IR Implementation

## Overview

This document describes the real Sea of Nodes intermediate representation implementation that replaces the previous mock implementation in the cppfort compiler.

## Architecture

### Core Components

#### 1. Node Classes

**NodeImpl** - Base implementation of IR nodes with:
- Type identification (Constant, BinaryOp, UnaryOp, Phi, Control, etc.)
- Unique ID for each node
- Input edges (operands/predecessors)
- Output edges (uses/successors)
- Value storage using std::any
- Pattern matching support

**Specialized Node Types:**
- `ConstantNode` - Integer constant values
- `BinaryOpNode` - Binary operations (Add, Sub, Mul, Div, And, Or, Xor, Shl, Shr, comparisons)
- `UnaryOpNode` - Unary operations (Neg, Not)
- `PhiNode` - SSA Phi nodes for control flow merge points
- `ControlNode` - Control flow nodes
- `RegionNode` - Control flow region with multiple predecessors
- `ProjectionNode` - Projects control/data from multi-output nodes

#### 2. Graph Class

**GraphImpl** - Manages the Sea of Nodes graph with:

**Node Management:**
- Node creation with automatic ID assignment
- Node deletion with edge cleanup
- Node map for fast lookup by ID
- Automatic constant deduplication (CSE for constants)

**Graph Queries:**
- Get all nodes
- Get root nodes (nodes with no inputs)
- Graph validation

**Analysis:**
- Dominance computation using iterative algorithm
- Dominance tree construction
- Immediate dominator queries
- Lowest common ancestor in dominance tree

**Scheduling:**
- Early scheduling (as close to inputs as possible)
- Late scheduling (as close to uses as possible)
- Depth-based ordering
- Schedule export for code generation

**Optimizations:**
- Constant folding
- Dead code elimination
- Common subexpression elimination
- Iterative optimization until fixed point

**Debug Support:**
- Graph dumping to text
- DOT graph visualization
- Validation of edge consistency

#### 3. Pattern Matching

**PatternMatcherImpl** - Supports pattern-based transformations:
- Custom matching predicates
- Replacement functions
- Graph-wide pattern search

#### 4. Lowering Passes

**LoweringPassImpl** - Implements optimization and lowering passes:
- Multiple patterns per pass
- Graph-wide application
- Changed flag tracking

#### 5. Target Lowering

**TargetLoweringImpl** - Target-specific code emission:

**C++ Emission:**
- Generates sequential C++ code
- SSA variables as C++ locals
- Type-safe int64_t operations
- Return last computed value

**MLIR Emission:**
- Generates MLIR arith dialect operations
- SSA form with % registers
- func.func and func.return
- Type annotations (i64)

## Implementation Details

### Edge Management

Edges are bidirectional:
- Each node maintains its inputs (operands)
- Each node maintains its outputs (uses)
- Adding/removing/replacing edges updates both directions automatically

This enables:
- Fast traversal in both directions
- Dead code detection (no outputs = dead, except control nodes)
- Use-def and def-use chains

### Dominance Analysis

Computes dominance relationships for control flow nodes using iterative fixed-point algorithm:

1. Initialize: start node dominates only itself
2. For all other nodes: all nodes dominate them initially
3. Iterate: `dom(n) = {n} ∪ (∩ dom(pred) for all preds)`
4. Repeat until fixed point
5. Compute immediate dominators from dominator sets
6. Build dominance tree

Used for:
- Global Code Motion legality checks
- Schedule placement
- Control flow optimization

### Scheduling Algorithm

Two-phase scheduling:

**Early Schedule:**
- Forward pass from inputs
- Place nodes as early as possible
- Depth = max(input_depths) + 1

**Late Schedule:**
- Backward pass from uses
- Place nodes as late as possible
- Minimize register pressure

Current implementation uses early schedule for code emission.

### Constant Folding

Pattern: BinaryOp(Constant, Constant) → Constant

Algorithm:
1. Find all binary operations
2. Check if both operands are constants
3. Evaluate operation at compile time
4. Create result constant
5. Replace all uses of operation with constant
6. Dead code elimination removes original operation

Handles:
- Arithmetic: +, -, *, /
- Bitwise: &, |, ^, <<, >>
- Comparisons: ==, !=, <, <=, >, >=

### Common Subexpression Elimination

Algorithm:
1. For each pair of nodes
2. Check if they are equivalent:
   - Same type
   - Same operation
   - Same operands
3. If equivalent, replace all uses of second with first
4. Dead code elimination removes redundant node

### Dead Code Elimination

Algorithm:
1. Mark nodes with no outputs as dead
2. Exception: control nodes are always live
3. Remove dead nodes
4. Repeat until no more dead nodes found

This is conservative - in a full implementation, we would also track control dependencies and side effects.

## Code Emission

### C++ Target

Generates sequential imperative C++ code:

```cpp
int64_t generated_function() {
    int64_t v1 = 15;
    int64_t v2 = 5;
    int64_t v3 = v1 - v2;  // 10
    int64_t v4 = 2;
    int64_t v5 = v3 * v4;  // 20
    return v5;
}
```

### MLIR Target

Generates MLIR arith dialect:

```mlir
module {
  func.func @generated_function() -> i64 {
    %v1 = arith.constant 15 : i64
    %v2 = arith.constant 5 : i64
    %v3 = arith.subi %v1, %v2 : i64
    %v4 = arith.constant 2 : i64
    %v5 = arith.muli %v3, %v4 : i64
    func.return %v5 : i64
  }
}
```

## Test Results

All tests passing:

- **Basic node creation**: Constants and operations properly linked
- **Constant folding**: 6 nodes reduced to 1 node
- **Dead code elimination**: 8 nodes reduced to 1 node (removed unused computation)
- **CSE**: Duplicate expressions merged
- **Scheduling**: Correct topological order by depth
- **Validation**: Edge consistency verified
- **DOT output**: Graph visualization generated
- **Code emission**: Both C++ and MLIR output correct
- **Full pipeline**: 9 nodes optimized to 1 node (constants folded, CSE applied, dead code removed)

## Performance Characteristics

**Time Complexity:**
- Node creation: O(1)
- Edge operations: O(degree)
- Dominance analysis: O(N^2) worst case, typically O(N log N)
- Scheduling: O(N + E) where E is edges
- Constant folding: O(N)
- CSE: O(N^2) (can be optimized with hashing)
- Dead code elimination: O(N * iterations)

**Space Complexity:**
- Graph: O(N + E)
- Dominance info: O(N^2) worst case, typically O(N)
- Schedule: O(N)

## Integration

The real IR is now built as `cppfort_ir_real` library and aliased as `cppfort_ir`. The mock implementation remains available as `cppfort_ir_mock` for compatibility.

Build artifacts:
- `libcppfort_ir_real.a` - Static library
- `test_sea_of_nodes` - Test executable

## Future Enhancements

Potential improvements:

1. **GCM (Global Code Motion)**
   - Full late scheduling implementation
   - Code motion across control flow
   - Register pressure estimation

2. **Enhanced Control Flow**
   - If/Then/Else regions
   - Loop regions
   - Exception handling

3. **Memory Operations**
   - Load/Store nodes
   - Memory dependencies
   - Alias analysis

4. **Type System**
   - Multiple types beyond int64_t
   - Type inference
   - Type conversions

5. **Advanced Optimizations**
   - Loop optimizations (unrolling, vectorization)
   - Inlining
   - Strength reduction
   - Algebraic simplifications

6. **More Targets**
   - LLVM IR emission
   - WebAssembly emission
   - Rust emission

7. **Debugging**
   - Source location tracking
   - Debug info preservation
   - Interactive graph visualization

## References

- Cliff Click's Sea of Nodes papers
- Simple compiler tutorial (Chapter 2 and beyond)
- MLIR documentation
- SSA book by Cytron et al.

## Files

- `/Users/jim/work/cppfort-ir-implementation/src/ir/sea_of_nodes_impl.h` - Header
- `/Users/jim/work/cppfort-ir-implementation/src/ir/sea_of_nodes_impl.cpp` - Implementation
- `/Users/jim/work/cppfort-ir-implementation/src/ir/test_sea_of_nodes.cpp` - Tests
- `/Users/jim/work/cppfort-ir-implementation/include/ir/sea_of_nodes.h` - Interface
- `/Users/jim/work/cppfort-ir-implementation/src/ir/CMakeLists.txt` - Build configuration
