# Multi-Index Subsumption Engine

**Status:** Design Phase
**Related:** [ADR-001](decisions/ADR-001-multi-index-subsumption-engine.md)

## Purpose

The subsumption engine provides unified query and projection capabilities across the Sea of Nodes IR, enabling:
- MLIR dialect conversions and pattern matching
- N-way graph transformations for optimization passes
- Hierarchical subsumption rules for type/control/data flow queries
- Node feature projection for analysis and code generation

## NOT a Generic Container

**Critical:** This is NOT Boost multi-index or a general-purpose container. This is a **compiler-specific subsumption rule engine** for graph query and transformation.

## Core Capabilities

### 1. Hash-Based Primary Index
Fast O(1) lookup by node ID as foundation for all queries.

### 2. Type Hierarchy Subsumption
Query nodes by type relationships:
```cpp
// All numeric types (subsumes int, float, double, etc.)
auto numerics = engine.whereType(subsumes<NumericType>());

// All arithmetic ops (AddNode, SubNode, MulNode, etc.)
auto arithOps = engine.whereType(subsumes<ArithmeticOp>());
```

### 3. Control Flow Queries
Query by CFG topology and dominance:
```cpp
// All nodes dominated by a loop header
auto inLoop = engine.whereCFG(dominatedBy(loop));

// Loop-invariant candidates (not dominated by loop but inputs are)
auto invariants = engine.whereCFG(notIn(loop))
                        .whereDataFlow(inputsDominate(loop));
```

### 4. Data Flow Projections
Query def-use chains and dependencies:
```cpp
// All loads that alias with a store
auto aliasedLoads = engine.whereMemory(aliasesWith(store));

// Find all uses of a phi node
auto phiUses = engine.whereDataFlow(usesNode(phi));
```

### 5. MLIR Pattern Matching
TableGen-style pattern matching for dialect conversions:
```cpp
// Match Sea of Nodes AddNode → MLIR arith.addi
Pattern addPattern = engine.createPattern()
    .match<AddNode>()
    .whereType(isIntegral())
    .rewrite([](AddNode* n) {
        return mlir::arith::AddIOp(n->lhs(), n->rhs());
    });
```

## Integration Points

**IMPORTANT:** Subsumption boundaries are NOT the same as band boundaries. The subsumption engine is **cross-band infrastructure** that queries concepts from all implemented bands simultaneously. See [Subsumption Boundaries vs Bands](subsumption-boundaries-vs-bands.md).

### Example: GCM Scheduling (Uses Band 2+3 Concepts)
GCM scheduling queries combine concepts from multiple bands:

```cpp
// Find candidates for loop hoisting
// - Loop depth: Band 2 concept
// - Dominator tree: Band 3 concept
// - Data flow: Band 2 concept
auto hoistCandidates = engine.query()
    .whereLoopDepth(lessThan(currentLoop->depth()))    // Band 2
    .whereDataFlow(allInputsAvailable(currentLoop->preheader()))  // Band 2
    .whereSchedule(earliestAt(preheader))              // Band 3
    .projectToSchedule();
```

### Example: Type-Based Optimization (Spans Multiple Bands)
Type queries can reference scheduling and control flow:

```cpp
// Find numeric operations that can be hoisted
// - Type: Band 4 concept
// - CFG: Band 2 concept
// - Schedule: Band 3 concept
auto compatible = engine.query()
    .whereType(subsumes<NumericType>())     // Band 4
    .whereCFG(dominatedBy(region))          // Band 2
    .whereSchedule(hoistable())             // Band 3
    .projectOptimizations();
```

### Example: MLIR Lowering (Cross-Band Projection)
MLIR integration queries ALL bands:

```cpp
// Lower operations to target dialect
// Query touches: types (Band 4), CFG (Band 2), scheduling (Band 3)
auto lowered = engine.query()
    .whereType(compatibleWith(targetDialect))          // Band 4
    .whereCFG(legalControlFlow(targetDialect))         // Band 2
    .whereSchedule(respectsDialectConstraints())       // Band 3
    .projectToMLIR(targetDialect);
```

**Key insight:** Each query creates dynamic subsumption boundaries based on optimization concerns, not band structure.

## Architecture

### Query Builder API
Fluent interface for composing multi-criteria queries:
```cpp
auto results = engine.query()
    .whereType<SpecificNode>()           // Type filter
    .whereCFG(condition)                  // Control flow filter
    .whereDataFlow(condition)             // Data flow filter
    .whereMLIR(condition)                 // MLIR-specific filter
    .project(projectionFn);               // Result transformation
```

### Rule Engine
Declarative rules for pattern matching and transformation:
```cpp
Rule constantFolding = engine.createRule()
    .match<AddNode>()
    .where([](AddNode* n) {
        return n->lhs()->isConstant() && n->rhs()->isConstant();
    })
    .rewrite([](AddNode* n) {
        auto lhs = static_cast<ConstantNode*>(n->lhs());
        auto rhs = static_cast<ConstantNode*>(n->rhs());
        return new ConstantNode(lhs->value() + rhs->value());
    });
```

### Subsumption Lattice
Type hierarchy and rule subsumption:
```
        Node
       /    \
    CFGNode  DataNode
    /    \      /   \
  Region  If  Phi  Constant
  /  \
Loop Start
```

## Implementation Phases

See [ADR-001](decisions/ADR-001-multi-index-subsumption-engine.md) for detailed implementation strategy.

**Current Phase:** Foundation design
**Next Phase:** Hash-based primary index implementation

## Performance Considerations

- **Primary index:** O(1) hash lookup
- **Type queries:** O(log n) via lattice traversal
- **CFG queries:** O(n) worst case, O(1) with dominator caching
- **Pattern matching:** O(n*m) where n=nodes, m=patterns

Optimization strategies:
- Cache dominator tree queries
- Incremental index updates
- Lazy projection computation
- Parallel query execution for independent filters

## Design Principles

1. **Query Composability** - Combine filters from different dimensions
2. **Lazy Evaluation** - Don't compute until projection
3. **Type Safety** - Compile-time guarantees where possible
4. **Extensibility** - New query types without core changes
5. **Performance** - Hash baseline, optimize hot paths

## Non-Goals

- NOT a generic multi-index container (not Boost)
- NOT a database query engine
- NOT for non-compiler use cases
- NOT a replacement for simple node traversal

## Current State

**Status:** Design/Planning
**Location:** `src/utils/multi_index.h` (currently contains placeholder 2D array)
**Action Required:** Remove misleading MultiIndex2D, implement subsumption foundation

**Critical Note:** Subsumption boundaries are query-driven and dynamic, NOT tied to band implementation phases. See [Subsumption Boundaries vs Bands](subsumption-boundaries-vs-bands.md) for clarification.

## References

- Band 3 GCM: `docs/band3_gcm.md`
- MLIR Patterns: https://mlir.llvm.org/docs/PatternRewriter/
- Sea of Nodes: Click & Palsberg papers
- Prolog Subsumption: Warren Abstract Machine (WAM)
