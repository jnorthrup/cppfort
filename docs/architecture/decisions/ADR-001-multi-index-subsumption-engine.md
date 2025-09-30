# ADR-001: Multi-Index Subsumption Engine for MLIR/Graph Projections

**Status:** Accepted
**Date:** 2025-09-30
**Architect:** Winston

## Context

**Critical divergence:** Cppfort's goals diverge from Simple compiler's Sea of Nodes approach. Simple emits to a single target (C++). Cppfort requires **n-way conversion** to multiple targets via TableGen pattern matching.

The cppfort project requires a sophisticated pattern matching and projection system for:

1. **N-WAY LOWERING (PRIMARY)** - Declarative pattern-based conversion from Sea of Nodes to multiple target IRs:
   - MLIR dialects (arith, memref, scf, vector, GPU)
   - Direct C emission
   - Direct C++ emission
   - Rust (via MLIR)
   - WASM (via MLIR)

2. **Pattern matching with constraints** - TableGen-style structural matching with subsumption checks:
   - Type hierarchy constraints (subsumes<IntegerType>)
   - CFG legality constraints (dominated_by, legal_control_flow)
   - Lifetime constraints (valid_borrow, escape_scope)

3. **Optimization queries (SECONDARY)** - Multi-dimensional graph queries for optimization passes

The current `src/utils/multi_index.h` contains a simple 2D array container (`MultiIndex2D`) which is NOT the intended multi-index system.

See [Divergence: Simple vs Cppfort N-Way](../divergence-son-simple-n-way.md) for full context.

## Decision

**Establish that "multi-index" refers to a pattern matching and subsumption engine for n-way lowering, NOT a generic container.**

**PRIMARY USE CASE:** Enable declarative pattern-based lowering from Sea of Nodes to multiple target IRs without imperative code explosion.

### Requirements

#### 1. Pattern Matching (PRIMARY)
TableGen-style declarative pattern matching for n-way lowering:
```cpp
// Pattern: Sea of Nodes Add → MLIR arith.addi
Pattern p = engine.createPattern()
    .match<AddNode>()
    .whereType(subsumes<IntegerType>())  // Type constraint
    .whereCFG(legal_control_flow())      // CFG constraint
    .rewrite([](AddNode* n) {
        return mlir::arith::AddIOp(n->lhs(), n->rhs());
    });

// Apply to all matching nodes
p.applyToGraph(graph);
```

#### 2. Constraint Checking (ESSENTIAL)
Subsumption-based constraint checking for semantic correctness:
- **Type hierarchy** - Subsumption lattice (e.g., subsumes<NumericType> matches int, float)
- **Control flow legality** - CFG constraints (dominated_by, legal_loop_structure)
- **Lifetime validity** - Borrow checking (valid_lifetime, escape_scope)
- **Memory safety** - Aliasing constraints (no_aliasing, same_alias_class)

#### 3. N-Way Target Dispatch
Support lowering to multiple targets from single source:
- **MLIR dialects** - arith, memref, scf, vector, GPU
- **Direct emission** - C, C++
- **Via MLIR** - Rust, WASM, GPU kernels
- **Constraint-based** - Different patterns per target

#### 4. Hash-Based Primary Index (FOUNDATION)
- O(1) node lookup by primary key (node ID, type, etc.)
- Baseline for pattern matching
- Fast single-dimension queries

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Multi-Index Subsumption Engine               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │ Hash Primary │   │Type Hierarchy│   │ CFG Topology │    │
│  │    Index     │   │    Index     │   │    Index     │    │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘    │
│         │                   │                   │            │
│         └───────────────────┴───────────────────┘            │
│                          │                                   │
│                   ┌──────▼─────┐                             │
│                   │ Subsumption │                            │
│                   │   Unifier   │                            │
│                   └──────┬──────┘                            │
│                          │                                   │
│         ┌────────────────┴────────────────┐                 │
│         ▼                                  ▼                 │
│  ┌──────────────┐                  ┌──────────────┐         │
│  │MLIR Pattern  │                  │Graph N-Way   │         │
│  │   Matcher    │                  │  Projection  │         │
│  └──────────────┘                  └──────────────┘         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Use Cases

#### MLIR Dialect Lowering
```cpp
// Find all arithmetic ops in loops that can lower to vector dialect
auto candidates = engine.query()
    .whereType<ArithmeticOp>()
    .whereLoopDepth(greaterThan(0))
    .whereVectorizable(true)
    .projectToMLIR(VectorDialect);
```

#### Sea of Nodes Optimization
```cpp
// Find loop-invariant code for hoisting (GCM Band 3)
auto invariants = engine.query()
    .whereControlFlow(notIn(loop))
    .whereDataFlow(allInputsDominateLoop(loop))
    .whereLoopDepth(lessThan(loop->getLoopDepth()))
    .projectToSchedule();
```

#### Subsumption Rules
```cpp
// Hierarchical type matching with unification
Rule loadStoreAlias = engine.createRule()
    .match(LoadNode, [](auto* load) { return load->alias(); })
    .match(StoreNode, [](auto* store) { return store->alias(); })
    .where(aliasesUnify())
    .action(insertAntiDependency());
```

## Implementation Strategy

### Phase 1: Foundation (Current)
- [ ] Remove misleading `MultiIndex2D` from `multi_index.h`
- [ ] Design core subsumption API
- [ ] Implement hash-based primary index
- [ ] Document intended usage patterns

### Phase 2: Type Hierarchy
- [ ] Build type subsumption lattice
- [ ] Implement type-based queries
- [ ] Integrate with Sea of Nodes type system

### Phase 3: CFG/Dominance
- [ ] Index nodes by dominator tree position
- [ ] Support loop-depth queries
- [ ] Enable control-flow projections

### Phase 4: MLIR Integration
- [ ] Map Sea of Nodes → MLIR operations
- [ ] Implement pattern matcher for TableGen integration
- [ ] Support dialect conversions

### Phase 5: Full N-Way Projections
- [ ] Arbitrary graph transformations
- [ ] Custom feature extractors
- [ ] Performance optimization

## Consequences

### Positive
- **Unified query interface** across all graph transformations
- **MLIR integration path** clearly defined
- **Subsumption semantics** enable sophisticated pattern matching
- **Performance** via multi-index optimization
- **Extensibility** through projection framework

### Negative
- **Complexity** - More sophisticated than simple container
- **Implementation effort** - Requires careful design across multiple phases
- **Learning curve** - Subsumption rules less familiar than standard containers

### Risks
- **Premature optimization** - Don't build until needed for MLIR/GCM
- **API churn** - Query interface may evolve as use cases emerge
- **Performance** - Multi-index overhead vs simple hash map trade-offs

## Related Decisions
- Band 3 GCM requires loop-invariant queries
- Band 4 Type System requires type hierarchy queries
- Future MLIR integration requires dialect pattern matching

**Important:** Subsumption boundaries are NOT tied to band implementation phases. A single subsumption query can span concepts from Bands 2, 3, 4 simultaneously. See [Subsumption Boundaries vs Bands](../subsumption-boundaries-vs-bands.md).

## References
- Sea of Nodes: Chapter 11 Global Code Motion (Band 3)
- MLIR Pattern Rewriting: https://mlir.llvm.org/docs/PatternRewriter/
- Prolog Subsumption: Logic programming unification
- Boost Multi-Index Containers: Design inspiration (but NOT the same)

## Notes

**CRITICAL CLARIFICATION:** This is NOT about adding Boost multi-index or a generic container library. This is about building a specialized subsumption rule engine for compiler graph queries with first-class hash lookup and hierarchical pattern matching.

The name "multi-index" reflects the ability to query nodes by multiple independent criteria simultaneously, NOT the Boost library of the same name.
