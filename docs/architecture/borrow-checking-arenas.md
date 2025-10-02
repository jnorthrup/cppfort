# Sea of Nodes Borrow Checking and Arena Allocators

**Status:** Design Phase (Band 5+ Integration)
**Clinical Goals:** Static memory safety guarantees without runtime overhead

## Overview

Sea of Nodes structure enables compile-time borrow checking through:
1. **Explicit lifetime edges** in the graph
2. **Dominance-based ownership** analysis
3. **Arena allocators** as "one-way drones" (no individual deallocation)
4. **Zero-cost abstractions** (Rust philosophy, C++ performance)

---

## SON-Based Borrow Assurances

### Clinical Definition

**Borrow checking** = Static verification that:
- References don't outlive referents
- Mutable borrows are exclusive
- Immutable borrows allow sharing
- No use-after-free, no double-free, no data races

### Sea of Nodes Advantages

Traditional borrow checkers (Rust's) operate on AST/HIR. Sea of Nodes operates on **dataflow graph**, providing:

1. **Explicit lifetime edges** - Ownership encoded as graph edges
2. **Dominance guarantees** - Lifetime relationships via dominator tree
3. **Escape analysis** - Graph structure reveals object lifetimes
4. **Region-based reasoning** - Control flow regions bound lifetimes

### Lifetime Encoding in Graph

#### Ownership Edges

```cpp
class BorrowNode : public Node {
    enum BorrowKind {
        SHARED,      // &T - immutable borrow
        MUTABLE,     // &mut T - exclusive borrow
        OWNED        // T - owned value (move semantics)
    };
private:
    BorrowKind _kind;
    Node* _referent;       // What is borrowed
    CFGNode* _scope_end;   // Where borrow expires
};
```

**Graph structure:**
```
AllocNode (owner)
    ↓ (owns edge)
BorrowNode (shared &T)
    ↓ (use edge)
LoadNode
    ↓ (dominates)
BorrowNode::_scope_end (CFG node where borrow dies)
```

### Borrow Checking Rules via Graph Queries

#### Rule 1: Exclusive Mutable Borrow

**Invariant:** No other borrows can coexist with `&mut T`.

```cpp
// Check via subsumption query
bool checkExclusiveMutability(BorrowNode* mutBorrow) {
    auto conflicts = engine.query()
        .whereReferent(mutBorrow->referent())
        .whereBorrowKind(anyBorrow())
        .whereCFG(overlapsLifetime(mutBorrow))
        .count();

    return conflicts == 0;  // Only mutBorrow exists
}
```

**Graph validation:**
```
Alloc (owner)
  ↓
BorrowMut (exclusive)
  ↓ uses
Store/Load operations
  ↓
ScopeEnd (borrow expires)
```

If another borrow exists in the dominated region → compile error.

#### Rule 2: Shared Borrows Don't Overlap with Mutable

**Invariant:** `&T` and `&mut T` cannot coexist.

```cpp
bool checkSharedMutExclusivity(BorrowNode* sharedBorrow) {
    auto mutConflicts = engine.query()
        .whereReferent(sharedBorrow->referent())
        .whereBorrowKind(MUTABLE)
        .whereCFG(overlapsLifetime(sharedBorrow))
        .projectViolations();

    return mutConflicts.empty();
}
```

**Graph structure ensures this via dominance:**
- If `&mut T` dominates uses, `&T` cannot be constructed
- If `&T` exists, `&mut T` creation is blocked (compile error)

#### Rule 3: No Use After Borrow Ends

**Invariant:** References invalid after scope exit.

```cpp
bool checkUseAfterBorrowEnd(BorrowNode* borrow) {
    auto usesAfterEnd = engine.query()
        .whereDataFlow(uses(borrow))
        .whereCFG(notDominatedBy(borrow->scopeEnd()))
        .projectInvalidUses();

    return usesAfterEnd.empty();
}
```

**Graph validation:**
```
BorrowNode (creates &T)
  ↓ dominates
Load/Store (valid uses)
  ↓ must dominate
ScopeEnd (borrow dies)
  ↓ does NOT dominate
Load (invalid use) ← COMPILE ERROR
```

### Integration with Band 2 Memory Model

Band 2 already tracks memory state via `MemProj` nodes. Extend with ownership:

```cpp
class MemProj : public Node {
    // Existing: memory slice projection
    // NEW: Track active borrows in this memory region
    std::vector<BorrowNode*> _active_borrows;

    bool canCreateMutableBorrow(Node* referent) {
        // Check if any borrows exist for this referent
        return std::none_of(_active_borrows.begin(), _active_borrows.end(),
            [&](BorrowNode* b) { return b->referent() == referent; });
    }
};
```

**Borrow checking becomes graph property checking:**
- Shared borrows = multiple MemProj edges from same source
- Mutable borrow = single MemProj edge (exclusive)
- Ownership transfer = MemProj edge rewired to new owner

---

## Arena Allocators: "One-Way Drone" Model

### Clinical Definition

**Arena allocator** = Bulk memory allocator where:
- Objects allocated sequentially in contiguous region
- No individual deallocation (one-way: allocate but not free)
- Entire arena deallocated at scope exit
- Zero fragmentation, minimal metadata overhead

### The "One-Way Drone" Metaphor

**Drone** = Allocation pointer that moves forward only:
```
Arena Start                        Arena End
[----allocated---->|<---free------|]
                   ^
                   drone (bump pointer)
```

**One-way** = Pointer never moves backward:
- Allocate: `drone += size` (forward)
- Deallocate: NO-OP (ignored)
- Reset: `drone = arena_start` (entire arena freed)

### Sea of Nodes Integration

#### ArenaNode: Region-Scoped Allocator

```cpp
class ArenaNode : public CFGNode {
private:
    size_t _arena_size;
    std::vector<NewNode*> _allocations;  // Objects in this arena
    CFGNode* _scope_end;                  // When arena deallocates

public:
    // Allocate in this arena (NewNode becomes child)
    NewNode* allocate(Type* type) {
        auto* alloc = new NewNode(this, type);
        alloc->setArena(this);  // Mark as arena-allocated
        _allocations.push_back(alloc);
        return alloc;
    }

    // Arena dies at scope exit (all allocations freed)
    void attachScopeEnd(CFGNode* scopeEnd) {
        _scope_end = scopeEnd;
        // All _allocations become invalid after scopeEnd
    }
};
```

#### Graph Structure

```
FunctionStart
  ↓
ArenaNode (allocator created)
  ↓ (dominates)
NewNode (allocation 1) ← arena-backed
  ↓
NewNode (allocation 2) ← arena-backed
  ↓
NewNode (allocation 3) ← arena-backed
  ↓
(work with allocations)
  ↓
ArenaNode::_scope_end (arena freed) ← ALL objects die here
  ↓
Return (no allocations valid)
```

**Key property:** Dominance guarantees no use-after-arena-free.

#### Borrow Checking with Arenas

```cpp
class ArenaBorrowNode : public BorrowNode {
    ArenaNode* _arena;  // Which arena owns this borrow

    bool isValid() const override {
        // Borrow valid only while arena alive
        return !cfg()->isDominatedBy(_arena->scopeEnd());
    }
};
```

**Validation via dominance:**
```cpp
bool checkArenaBorrowValidity(ArenaBorrowNode* borrow) {
    auto usesAfterArenaEnd = engine.query()
        .whereDataFlow(uses(borrow))
        .whereCFG(postDominates(borrow->arena()->scopeEnd()))
        .projectInvalidUses();

    if (!usesAfterArenaEnd.empty()) {
        reportError("Use of arena-allocated object after arena deallocation");
        return false;
    }
    return true;
}
```

### Arena Allocation Strategy

#### Fast Path: Bump Allocation

```cpp
void* Arena::allocate(size_t size, size_t align) {
    // One-way drone: pointer only moves forward
    void* ptr = (void*)((_bump_ptr + align - 1) & ~(align - 1));
    _bump_ptr = (char*)ptr + size;

    if (_bump_ptr > _arena_end) {
        expandArena();  // Rare: grow arena
    }

    return ptr;  // NO metadata, NO free list
}

void Arena::deallocate(void* ptr) {
    // NO-OP: one-way drone never goes backward
    // Individual objects cannot be freed
}

void Arena::reset() {
    // Bulk deallocation: drone returns to start
    _bump_ptr = _arena_start;
    // All objects dead in one instruction
}
```

**Performance characteristics:**
- Allocate: 3-4 instructions (vs malloc: 100+ instructions)
- Deallocate: 0 instructions (NO-OP)
- Reset: 1 instruction (bulk free)

#### Graph Optimization

Sea of Nodes can **prove arena safety** at compile time:

```cpp
// Escape analysis + arena allocation
auto arenaAllocatable = engine.query()
    .whereType<NewNode>()
    .whereEscape(doesNotEscape(functionScope))
    .whereCFG(dominatedBy(arenaNode))
    .projectToArena(arenaNode);

// Convert heap allocations to arena allocations
arenaAllocatable.forEach([](NewNode* alloc) {
    alloc->lowerToArena();  // Zero-cost: no malloc, no free
});
```

**Result:** Entire class of allocations becomes **stack-like** performance.

---

## Integration with Upcoming Bands

### Band 5: Escape Analysis

**Goal:** Determine which allocations can use arena allocators.

```cpp
// Escape analysis via subsumption
auto localAllocations = engine.query()
    .whereType<NewNode>()
    .whereEscape([](NewNode* n) {
        // Object doesn't escape if:
        // 1. Not stored to heap
        // 2. Not returned from function
        // 3. Not captured by closure
        // 4. All references die before function exit
        return !n->escapesFunction();
    })
    .projectToLocalArena();

// Lower to arena allocation
localAllocations.forEach([](NewNode* alloc) {
    alloc->setAllocator(localArena);
});
```

**Subsumption benefit:** Bulk allocation strategy decision.

### Band 5+: Region-Based Memory Management

**Goal:** Hierarchical arenas matching control flow regions.

```cpp
class RegionNode : public CFGNode {
    ArenaNode* _region_arena;  // Each region gets own arena

    void enterRegion() {
        _region_arena = new ArenaNode(estimateRegionSize());
    }

    void exitRegion() {
        _region_arena->reset();  // Bulk free entire region
    }
};
```

**Graph structure:**
```
FunctionStart
  ↓
Region1 (arena1)
  ↓ allocations in arena1
  If condition
    ↓ true
    Region2 (arena2) ← nested arena
      ↓ allocations in arena2
      work...
    ↓ Region2 exit (arena2.reset())
  ↓
  Region1 continues (arena1 still alive)
↓
Region1 exit (arena1.reset())
```

**Nested arenas = nested scopes** (matches control flow hierarchy).

---

## Simplistic "E.g. One-Way Drone" Arena Examples

### Example 1: Parser Allocator

**Use case:** Parsing AST nodes - allocated during parse, freed in bulk after.

```cpp
class Parser {
    ArenaNode* _parse_arena;

    ASTNode* parseExpression() {
        // All AST nodes allocated in parse arena
        auto* node = _parse_arena->allocate<ASTNode>();
        node->left = parseSubExpr();   // Also arena-allocated
        node->right = parseSubExpr();  // Also arena-allocated
        return node;
    }

    void parseFile() {
        _parse_arena = new ArenaNode(1_MB);  // Pre-size
        auto* ast = parseExpression();
        processAST(ast);
        _parse_arena->reset();  // Entire AST freed instantly
    }
};
```

**Graph encoding:**
```
ParseArena (created)
  ↓
NewNode<ASTNode> (expr1)
  ↓
NewNode<ASTNode> (expr2)
  ↓
NewNode<ASTNode> (expr3)
  ↓
ProcessAST (reads nodes)
  ↓
ParseArena::reset() ← ALL AST nodes freed
```

**Borrow checking:** Graph ensures no AST uses after `reset()`.

### Example 2: Request Handler Arena

**Use case:** HTTP request processing - allocate request data, free after response.

```cpp
class RequestHandler {
    void handleRequest(Request* req) {
        ArenaNode* reqArena = new ArenaNode(64_KB);

        // All request processing in arena
        auto* parsed = parseRequest(req, reqArena);
        auto* response = processRequest(parsed, reqArena);
        sendResponse(response);

        reqArena->reset();  // Everything freed at once
    }
};
```

**Graph encoding:**
```
RequestArena (created)
  ↓
ParseData (arena-allocated)
  ↓
IntermediateData (arena-allocated)
  ↓
Response (arena-allocated)
  ↓
SendResponse (transmits, doesn't retain)
  ↓
RequestArena::reset() ← ALL request data freed
```

**Escape analysis:** Graph proves response data doesn't escape, safe to arena-allocate.

### Example 3: Temporary Graph Construction

**Use case:** Build intermediate graph, extract result, discard graph.

```cpp
class Optimizer {
    Node* optimizeSubgraph(Node* root) {
        ArenaNode* tempArena = new ArenaNode(1_MB);

        // Build temporary graph in arena
        auto* tempGraph = cloneToArena(root, tempArena);
        applyOptimizations(tempGraph);
        auto* result = extractOptimizedCode(tempGraph);

        tempArena->reset();  // Discard temporary graph
        return result;
    }
};
```

**Graph encoding:**
```
TempArena (created)
  ↓
ClonedNodes (arena-allocated temporary graph)
  ↓
OptimizedNodes (arena-allocated)
  ↓
ExtractResult (copies result out of arena)
  ↓
TempArena::reset() ← Temporary graph freed
  ↓
Return result (NOT arena-allocated)
```

**Critical:** `extractResult()` must **copy** data out of arena before reset.

---

## Clinical Goals

### 1. Zero-Cost Memory Safety

**Goal:** Rust-like safety without runtime overhead.

**Mechanism:**
- Borrow checking = static graph analysis (compile time)
- Lifetime validation = dominance property checking (compile time)
- Arena allocation = bump pointer (3-4 instructions)
- Arena deallocation = NO-OP or single pointer reset

**Result:** Memory safety is **free** (no runtime checks, no GC).

### 2. Explicit Lifetime Boundaries

**Goal:** Programmer control over allocation strategy.

**Mechanism:**
```cpp
// Explicit arena scope in source language
arena myArena {
    auto* obj1 = new(myArena) Object();
    auto* obj2 = new(myArena) Object();
    process(obj1, obj2);
} // myArena.reset() here - both objects freed
```

**Graph encoding:**
```
ArenaNode (myArena created)
  ↓
NewNode(myArena, Object) ← arena-allocated
  ↓
NewNode(myArena, Object) ← arena-allocated
  ↓
Process (uses objects)
  ↓
ScopeEnd (myArena.reset()) ← Both freed
```

### 3. Subsumption-Based Allocation Strategy

**Goal:** Bulk allocation decisions via type/escape/lifetime queries.

**Mechanism:**
```cpp
// Identify candidates for arena allocation
auto arenaCandidates = engine.query()
    .whereType<NewNode>()
    .whereEscape(local())
    .whereLifetime(boundedBy(currentRegion))
    .groupBy(lifetimeRegion());

// Convert to arena allocations in bulk
arenaCandidates.forEach([](auto& regionGroup) {
    auto* arena = createArenaForRegion(regionGroup.region());
    regionGroup.allocations().lowerToArena(arena);
});
```

**Coinserter pattern:** Group allocations by lifetime region, assign arena per group.

### 4. Linear Type Approximation

**Goal:** Track move-only types (no copy semantics).

**Mechanism:**
```cpp
class MoveOnlyNode : public Node {
    bool _consumed = false;

    void markConsumed() {
        assert(!_consumed);
        _consumed = true;
    }

    bool checkLinearUse() {
        auto uses = engine.query()
            .whereDataFlow(uses(this))
            .count();
        return uses <= 1;  // At most one use
    }
};
```

**Graph validation:** If move-only object used twice → compile error.

---

## Implementation Roadmap

### Phase 1: Borrow Node Infrastructure (Band 5)
- [ ] Implement `BorrowNode` with shared/mutable/owned variants
- [ ] Add lifetime edges to graph
- [ ] Integrate with Band 2 memory model

### Phase 2: Dominance-Based Validation (Band 5)
- [ ] Implement borrow checking rules via subsumption queries
- [ ] Add compile-time lifetime validation
- [ ] Error reporting for borrow violations

### Phase 3: Arena Allocator Integration (Band 5+)
- [ ] Implement `ArenaNode` as CFG node
- [ ] Add bump allocation strategy
- [ ] Integrate with escape analysis

### Phase 4: Region-Based Management (Band 5+)
- [ ] Hierarchical arenas matching control flow regions
- [ ] Nested arena lifetime tracking
- [ ] Bulk deallocation at scope exit

### Phase 5: Optimization (Band 5+)
- [ ] Convert provably-local allocations to arena allocations
- [ ] Eliminate redundant lifetime checks
- [ ] Specialize arena sizes based on profiling

---

## Validation Strategy

### Static Guarantees

**Proven at compile time:**
1. No use-after-free (dominance property)
2. No double-free (arena bulk deallocation only)
3. No data races (mutable borrow exclusivity)
4. No dangling pointers (lifetime dominance)

### Runtime Checks (Debug Mode)

**Inserted only in debug builds:**
```cpp
#ifndef NDEBUG
    if (borrow->isExpired()) {
        reportLifetimeViolation(borrow);
    }
#endif
```

**Removed in release builds** (zero-cost abstraction).

---

## Comparison with Existing Systems

| Feature | Rust | C++ RAII | Sea of Nodes |
|---------|------|----------|--------------|
| **Borrow checking** | Yes (HIR) | No | Yes (graph) |
| **Arena allocators** | Manual | Manual | Automatic |
| **Zero-cost safety** | Yes | Partial | Yes |
| **Escape analysis** | Yes | No | Yes (Band 5+) |
| **Graph-based validation** | No | No | Yes |
| **MLIR integration** | No | No | Yes |

**Advantage:** Graph structure enables **compiler-driven** arena allocation decisions.

---

## Conclusion

Sea of Nodes enables:

1. **Borrow checking** via graph dominance properties (Rust-like safety)
2. **Arena allocators** as "one-way drones" (C-like performance)
3. **Subsumption-based allocation strategy** (bulk optimization)
4. **Zero-cost abstractions** (compile-time validation, no runtime overhead)

**Clinical result:** Memory safety without GC, without runtime checks, with explicit programmer control over allocation strategy.

**Next step:** Implement escape analysis (Band 5) to enable automatic arena allocation inference.
