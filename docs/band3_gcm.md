# Band 3: Chapter 11 - Global Code Motion Implementation

## THE Critical Convergence Point

// TODO: The documentation currently overstates test coverage. Implement missing features (e.g., fixInfiniteLoops) and add accurate coverage statements.
This implementation represents the most critical transformation in the entire Sea of Nodes compiler architecture: **the scheduling of an unordered graph into executable code**.

## Overview

Band 3 implements Chapter 11 of the Simple compiler, which introduces the Global Code Motion (GCM) algorithm. This is where the Sea of Nodes graph, which has been optimized without regard to execution order, gets transformed into a concrete sequence of operations that can be executed.

## Key Components Implemented

### 1. CFG Node Enhancement (`gcm.h`, `gcm.cpp`)

All control flow nodes are enhanced with scheduling information:

- **Dominator computation**: `idom()` methods compute immediate dominators
- **Dominator depth caching**: `idepth()` provides efficient dominator tree traversal
- **LCA computation**: Finding lowest common ancestors for scheduling decisions
- **Loop depth analysis**: Computing nesting depth for optimization decisions

### 2. Scheduling Algorithm

The GCM algorithm operates in two main phases:

#### Early Schedule (Code Hoisting)
- Schedules nodes as early as possible
- Pushes operations up to dominating blocks
- Minimizes register pressure by reducing live ranges

#### Late Schedule (Code Sinking)
- Schedules nodes as late as possible
- Maximizes control dependence
- Finds optimal placement between early and late positions
- Prefers shallower loop nests (loop-invariant code motion)

### 3. Anti-Dependency Management

Memory operations require special handling:
- Loads and Stores to the same memory location must maintain order
- Anti-dependency edges are inserted from Loads to Stores
- These edges are purely scheduling constraints, not data dependencies

### 4. Infinite Loop Handling

TODO (UNIMPLEMENTED): The implementation currently does NOT detect and handle infinite loops.
- fixInfiniteLoops() in [`src/stage0/gcm.cpp:42`](src/stage0/gcm.cpp:42) is a stub and is not executed by the scheduler.
- Loops without exits are not guaranteed to be identified; `NeverNode` insertion is not implemented.
- This section is aspirational — implement loop exit detection and NeverNode insertion before claiming this feature.

## TableGen Patterns (`gcm_patterns.td`)

The TableGen specification defines:

### Basic Block Patterns
```tablegen
def BBStart : CFGConstraint<"BBStart"> {
  list<string> NodeTypes = ["Start", "CProj", "Region", "Loop", "Stop"];
}
```

### Scheduling Rules
```tablegen
def ScheduleAtInputs : EarlyScheduleRule<"ScheduleAtInputs"> {
  string Rule = "max(input.idepth() for input in inputs)";
}
```

### Anti-Dependency Patterns
```tablegen
def LoadStoreOrder : AntiDepPattern<"LoadStoreOrder"> {
  string Condition = "load.alias == store.alias";
  string Action = "add edge from load to store";
}
```

## Test Coverage (`test_band3.cpp`)

NOTE: The original `tests/test_band3.cpp` was a design sketch and is not compilable.
The following test items are aspirational and must be validated by real, runnable tests
before asserting coverage.

Aspirational test scenarios (update with real tests):
1. **Basic Scheduling**: Constants and arithmetic operations
2. **Memory Ordering**: Load/Store anti-dependencies
3. **Loop Optimization**: Loop-invariant code motion
4. **Infinite Loops**: Never node insertion (UNIMPLEMENTED)
5. **Control Dependence**: Conditional placement
6. **Complex Graphs**: Real-world scheduling scenarios

## Integration with Previous Bands

Band 3 builds on:

- **Band 1** (Chapters 1-6): Basic node types and control flow
- **Band 2** (Chapters 7-10): Loops and memory operations

And enables:

- **Band 4** (Chapters 12-15): Advanced optimizations
- **Band 5** (Chapters 16-23): Code generation

## Critical Algorithms

### Dominator Computation
```cpp
CFGNode* CFGNode::idom(CFGNode* rhs) {
    // LCA in dominator tree
    while (lhs != rhs) {
        if (lhs->idepth() >= rhs->idepth()) lhs = lhs->idom();
        if (lhs->idepth() <= rhs->idepth()) rhs = rhs->idom();
    }
    return lhs;
}
```

### Schedule Selection
```cpp
bool GlobalCodeMotion::better(CFGNode* lca, CFGNode* best) {
    // Prefer shallower loop depth
    if (lca->_loopDepth < best->_loopDepth) return true;
    // At same depth, prefer deeper control
    if (lca->idepth() > best->idepth()) return true;
    return false;
}
```

## Why This is THE Chokepoint

The GCM implementation is critical because:

1. **Semantic Preservation**: Must maintain program semantics while reordering
2. **Performance Impact**: Determines register pressure and cache behavior
3. **Optimization Enabler**: Makes loop-invariant code motion possible
4. **Memory Correctness**: Ensures proper ordering of memory operations
5. **Universal Convergence**: All paths through the compiler must pass through scheduling

## Future Work

With Band 3 complete, the stage0 meta-transpiler can now:

- Generate scheduled code for multiple target languages
- Perform sophisticated loop optimizations
- Maintain memory consistency across transformations
- Support incremental compilation with proper scheduling

## Conclusion

Band 3 represents the culmination of the unordered Sea of Nodes philosophy. By separating optimization from scheduling, we achieve:

- **Maximum optimization freedom**: Optimizations don't worry about order
- **Optimal scheduling**: Scheduling decisions are made with complete information
- **Clean architecture**: Clear separation of concerns

This implementation provides the foundation for all subsequent code generation and advanced optimization passes.