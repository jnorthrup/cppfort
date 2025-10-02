# ADR-002: Arena Allocators and Graph-Based Borrow Checking

**Status:** Accepted (Design Phase)
**Date:** 2025-09-30
**Architect:** Winston
**Related:** ADR-001 (Subsumption Engine)

## Context

Modern systems programming requires:
1. **Memory safety** without garbage collection overhead
2. **Zero-cost abstractions** (Rust philosophy, C++ performance)
3. **Explicit allocation control** (programmer understands costs)
4. **Bulk optimization** (subsumption-based allocation strategies)

Traditional approaches:
- **C/C++:** Fast but unsafe (use-after-free, double-free, data races)
- **Rust:** Safe but complex (lifetime annotations, borrow checker learning curve)
- **Java/Go/C#:** Safe but GC pauses (unacceptable for real-time systems)

## Decision

**Adopt Sea of Nodes graph structure for borrow checking and arena allocation.**

### Component 1: Graph-Based Borrow Checking

Encode ownership and lifetimes as graph edges:

```cpp
class BorrowNode : public Node {
    enum BorrowKind { SHARED, MUTABLE, OWNED };
    Node* _referent;      // What is borrowed
    CFGNode* _scope_end;  // Where borrow expires
};
```

**Validation via dominance queries:**
```cpp
// Ensure no use after borrow ends
auto invalidUses = engine.query()
    .whereDataFlow(uses(borrow))
    .whereCFG(notDominatedBy(borrow->scopeEnd()))
    .projectViolations();

if (!invalidUses.empty()) {
    reportError("Use after borrow expiration");
}
```

### Component 2: Arena Allocators as "One-Way Drones"

**One-way drone** = Allocation pointer moves forward only:

```cpp
class ArenaNode : public CFGNode {
    void* allocate(size_t size) {
        void* ptr = _bump_ptr;
        _bump_ptr += size;  // Move forward
        return ptr;
    }

    void deallocate(void*) {
        // NO-OP: one-way, no backward movement
    }

    void reset() {
        _bump_ptr = _arena_start;  // Bulk free
    }
};
```

**Performance:**
- Allocate: 3-4 instructions (vs malloc: 100+)
- Deallocate: 0 instructions (NO-OP)
- Bulk free: 1 instruction

### Component 3: Subsumption-Based Allocation Strategy

Use subsumption engine to **bulk-decide allocation strategies:**

```cpp
// Identify arena-allocatable objects
auto arenaCandidates = engine.query()
    .whereType<NewNode>()
    .whereEscape(doesNotEscape(functionScope))
    .whereLifetime(boundedBy(currentRegion))
    .groupBy(lifetimeRegion());

// Assign arena per lifetime region
arenaCandidates.forEach([](auto& group) {
    group.allocations().lowerToArena(group.arena());
});
```

**Coinserter pattern:** Each lifetime region gets dedicated arena.

## Rationale

### Why Graph-Based Borrow Checking?

1. **Explicit dataflow** - Ownership edges visible in IR
2. **Dominance guarantees** - Lifetime relationships via CFG structure
3. **Compile-time validation** - No runtime overhead
4. **MLIR integration** - Graph structure maps to MLIR dialects
5. **Subsumption queries** - Bulk validation of borrow rules

### Why Arena Allocators?

1. **Performance** - 30x faster than malloc for temporary allocations
2. **Deterministic** - No GC pauses, predictable latency
3. **Bulk deallocation** - Entire lifetime regions freed at once
4. **Natural fit** - Control flow regions = arena lifetimes
5. **Zero fragmentation** - Sequential bump allocation

### Why "One-Way Drone" Model?

**Simplicity:** Allocation pointer never moves backward:
- No free list maintenance
- No coalescing logic
- No fragmentation handling
- One pointer arithmetic operation

**Safety:** No individual deallocation means:
- No double-free possible
- No use-after-free within arena (validated by graph)
- Bulk reset is trivially safe

## Implementation Strategy

### Phase 1: Escape Analysis (Band 5)

Determine which allocations can be arena-backed:

```cpp
auto escapingAllocs = engine.query()
    .whereType<NewNode>()
    .whereEscape(escapesFunction())
    .projectHeapAllocations();

auto localAllocs = engine.query()
    .whereType<NewNode>()
    .whereEscape(doesNotEscape())
    .projectArenaAllocations();
```

### Phase 2: Borrow Node Infrastructure (Band 5)

Add lifetime edges to graph:

```cpp
class BorrowNode : public Node {
    void attachScopeEnd(CFGNode* end) {
        _scope_end = end;
        // Validate all uses dominated by end
    }
};
```

### Phase 3: Arena Node Integration (Band 5+)

Hierarchical arenas matching control flow:

```cpp
class RegionNode : public CFGNode {
    ArenaNode* _region_arena;

    void enterRegion() {
        _region_arena = createArena();
    }

    void exitRegion() {
        _region_arena->reset();  // Bulk free
    }
};
```

### Phase 4: Optimization (Band 5+)

Convert provably-local allocations to arenas:

```cpp
// Subsumption-based optimization
localAllocs.forEach([](NewNode* alloc) {
    if (alloc->lifetime().boundedBy(currentRegion)) {
        alloc->lowerToArena(currentRegion->arena());
    }
});
```

## Consequences

### Positive

1. **Memory safety without GC** - Compile-time validation, no runtime overhead
2. **Explicit control** - Programmer chooses allocation strategy
3. **Predictable performance** - No GC pauses, deterministic allocation
4. **Bulk optimization** - Subsumption enables region-wide decisions
5. **Zero-cost abstractions** - Safety is free (compile-time only)
6. **MLIR integration** - Graph structure maps to MLIR lifetime attributes

### Negative

1. **Complexity** - Borrow checking adds compiler complexity
2. **Learning curve** - Programmers must understand lifetime semantics
3. **Implementation effort** - Escape analysis is non-trivial
4. **Diagnostic quality** - Error messages must be clear (Rust lesson learned)

### Risks

1. **Premature implementation** - Need Band 5 escape analysis first
2. **API stability** - Borrow syntax may evolve based on usage
3. **Performance validation** - Must benchmark arena benefits
4. **Compatibility** - Integration with existing C/C++ code requires FFI strategy

## Comparison with Alternatives

| Approach | Safety | Performance | Complexity | Runtime Overhead |
|----------|--------|-------------|------------|------------------|
| **C/C++ (raw pointers)** | ❌ None | ✅ Fast | ✅ Simple | None |
| **Rust borrow checker** | ✅ Strong | ✅ Fast | ❌ Complex | None |
| **Java/C# GC** | ✅ Strong | ⚠️ Pauses | ✅ Simple | High |
| **C++ smart pointers** | ⚠️ Partial | ⚠️ Overhead | ⚠️ Medium | RC/atomic ops |
| **SON arenas + borrow** | ✅ Strong | ✅ Fast | ⚠️ Medium | None |

**Sweet spot:** Strong safety + C performance + reasonable complexity.

## Integration with Subsumption Engine

Arena allocation is a **coinserter pattern** - partition by lifetime:

```cpp
// Lifetime-based partitioning
auto byLifetime = engine.query()
    .whereType<NewNode>()
    .groupBy(lifetimeRegion());

// Each partition gets dedicated arena
byLifetime.forEach([](auto& regionGroup) {
    auto* arena = createArenaForRegion(regionGroup.region());
    regionGroup.allocations().lowerToArena(arena);
});
```

**Orthogonal consolidation:** All allocations with same lifetime → single arena.

## Example: Request Handler

**Source code:**
```cpp
arena requestArena {
    auto* parsed = parseRequest(req);
    auto* response = processRequest(parsed);
    sendResponse(response);
} // requestArena.reset() - all freed
```

**Graph structure:**
```
ArenaNode (requestArena)
  ↓
NewNode (parsed) ← arena-allocated
  ↓
NewNode (response) ← arena-allocated
  ↓
SendResponse (reads response)
  ↓
ArenaNode::reset() ← Bulk free
```

**Borrow validation:** Graph proves no uses after arena reset.

## Success Criteria

1. **Static safety** - All borrow violations caught at compile time
2. **Zero overhead** - No runtime checks in release builds
3. **Performance win** - Arena allocation 10x+ faster than malloc
4. **Programmer control** - Explicit arena scopes in source language
5. **Bulk optimization** - Subsumption enables region-wide allocation strategy

## Related Decisions

- **ADR-001:** Subsumption engine provides bulk allocation queries
- **Band 5:** Escape analysis determines arena eligibility
- **Band 3:** Dominance analysis validates lifetime boundaries
- **Band 2:** Memory model extended with ownership edges

## References

- Rust borrow checker: https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html
- Arena allocators: Hanson "Fast Allocation and Deallocation of Memory Based on Object Lifetimes" (1990)
- Linear types: Wadler "Linear Types Can Change the World!" (1990)
- Escape analysis: Choi et al. "Escape Analysis for Java" (1999)

## Notes

**Clinical goals:**
1. Memory safety without runtime overhead
2. Explicit lifetime boundaries (programmer control)
3. Subsumption-based bulk allocation strategy
4. "One-way drone" arenas (bump allocation, bulk deallocation)

**Implementation priority:** After Band 5 escape analysis is complete.
