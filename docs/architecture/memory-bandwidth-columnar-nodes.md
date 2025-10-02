# Conquering Memory Bandwidth: Columnar Sea of Nodes

**Core Problem:** Sea of Nodes nodes are pointer-rich structures with bidirectional edges. Each node traversal = cache miss = memory bandwidth bottleneck.

**Question:** Can columnar layout reduce bandwidth while preserving graph semantics? What are the drag factors?

---

## The Memory Bandwidth Problem

### Traditional Node Layout (Pointer-Based)

```cpp
class Node {
    int _nid;                          // 4 bytes
    Type* _type;                       // 8 bytes (pointer)
    std::vector<Node*> _inputs;        // 24 bytes (vector overhead)
    std::vector<Node*> _outputs;       // 24 bytes (vector overhead)
    // Total: 60+ bytes per node
};
```

**Memory access pattern for graph traversal:**
```
Node A (cache miss) → read 60 bytes
  ↓ follow _inputs[0] pointer
Node B (cache miss) → read 60 bytes
  ↓ follow _inputs[0] pointer
Node C (cache miss) → read 60 bytes
```

**Bandwidth waste:**
- Each node access loads entire 64-byte cache line
- Most fields unused during specific operations
- Pointer chasing = random memory access = bandwidth killer

---

## Columnar Layout: Structure of Arrays (SoA)

### Concept: Separate by Field

Instead of `Array[Node]`, use `Struct{Array[field1], Array[field2], ...}`:

```cpp
struct ColumnarNodes {
    std::vector<int> node_ids;           // Packed IDs
    std::vector<Type*> types;            // Type pointers
    std::vector<uint32_t> input_offsets; // Start offset in inputs array
    std::vector<uint32_t> input_counts;  // Count of inputs
    std::vector<Node*> inputs_packed;    // All inputs contiguous
    std::vector<uint32_t> output_offsets;
    std::vector<uint32_t> output_counts;
    std::vector<Node*> outputs_packed;
};
```

### Memory Layout Comparison

**Pointer-based (AoS - Array of Structures):**
```
Memory: [NodeA_all_fields][NodeB_all_fields][NodeC_all_fields]...
Access:  ^^^^^^^^^^^^^^^^^ cache line (64 bytes, mostly wasted)
```

**Columnar (SoA - Structure of Arrays):**
```
Memory: [IDs: A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P][Types: ...][Inputs: ...]
Access:  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ cache line (16 IDs, all used!)
```

**Bandwidth win:** 16x better utilization when accessing only IDs.

---

## Columnar Rules: When They Win

### Rule 1: Sequential Scans

**Operation:** Find all nodes of specific type.

**Pointer-based:**
```cpp
for (Node* n : all_nodes) {  // Random access, cache misses
    if (n->type() == target) {
        process(n);
    }
}
// Bandwidth: 60 bytes read per node checked
```

**Columnar:**
```cpp
for (size_t i = 0; i < node_ids.size(); i++) {
    if (types[i] == target) {  // Sequential access, cache hits
        process(i);
    }
}
// Bandwidth: 8 bytes read per node checked (just type pointer)
```

**Win:** 7.5x less bandwidth for type-based queries.

### Rule 2: Bulk Property Extraction

**Operation:** Extract loop depths for all CFG nodes.

**Pointer-based:**
```cpp
for (CFGNode* cfg : cfg_nodes) {
    depths.push_back(cfg->loopDepth());  // Pointer chase per node
}
// Bandwidth: Full node load per access
```

**Columnar:**
```cpp
// Pre-computed column
std::vector<int> loop_depths;  // Already extracted

// Or compute in bulk:
for (size_t i : cfg_node_indices) {
    depths[i] = computeLoopDepth(i);  // Sequential access
}
// Bandwidth: Only loop-depth-relevant fields accessed
```

**Win:** Amortize graph traversal over all nodes.

### Rule 3: Subsumption Queries

**Operation:** Find all arithmetic nodes dominated by loop.

**Columnar advantage:**
```cpp
// Bitmap-based filtering (extreme columnar)
std::bitset<MAX_NODES> is_arithmetic;
std::bitset<MAX_NODES> is_dominated;

// Compute once:
is_arithmetic = computeArithmeticMask();
is_dominated = computeDominatorMask(loop);

// Query = bitwise AND (SIMD-friendly)
auto result = is_arithmetic & is_dominated;
// Bandwidth: 2 bits per node (512x compression!)
```

**Win:** Queries become bitwise operations on packed bitmaps.

---

## Drag Factors: When Columnar Loses

### Drag Factor 1: Random Access Patterns

**Operation:** Follow def-use chains.

**Pointer-based:**
```cpp
Node* current = start;
while (current) {
    current = current->input(0);  // Direct pointer follow
}
// Bandwidth: High, but unavoidable
```

**Columnar:**
```cpp
size_t idx = start_idx;
while (idx != NULL_INDEX) {
    size_t offset = input_offsets[idx];      // Indirection 1
    idx = inputs_packed[offset];             // Indirection 2
}
// Bandwidth: WORSE - double indirection per step
```

**Loss:** Columnar adds indirection overhead for pointer chasing.

### Drag Factor 2: Sparse Updates

**Operation:** Modify single node's inputs.

**Pointer-based:**
```cpp
node->setInput(2, new_input);  // Update one vector
// Bandwidth: One cache line write
```

**Columnar:**
```cpp
// Must update:
// 1. input_counts[idx] if resizing
// 2. Shift entire inputs_packed array if growing
// 3. Update all downstream input_offsets
// Bandwidth: Potentially ENTIRE column rewrite
```

**Loss:** Sparse updates = bulk column rewrite (disaster).

### Drag Factor 3: Mixed-Width Access

**Operation:** Access both node type and inputs.

**Pointer-based:**
```cpp
Type* t = node->type();
Node* inp = node->input(0);
// Bandwidth: One cache line (both in same struct)
```

**Columnar:**
```cpp
Type* t = types[idx];                    // Access column 1
size_t offset = input_offsets[idx];      // Access column 2
Node* inp = inputs_packed[offset];       // Access column 3
// Bandwidth: Three separate cache lines
```

**Loss:** Cross-column access pattern = scattered reads.

### Drag Factor 4: Memory Allocation Overhead

**Pointer-based:**
```cpp
new Node();  // Single malloc
```

**Columnar:**
```cpp
// Must resize ALL columns:
node_ids.push_back(id);
types.push_back(type);
input_offsets.push_back(offset);
// etc... (7+ vector resizes)
// Potential reallocation storm
```

**Loss:** Node creation = multiple vector resizes.

---

## Hybrid Strategy: Conquer Bandwidth Without Drag

### Strategy 1: Hot/Cold Splitting

**Observation:** Not all fields accessed with same frequency.

**Split by access pattern:**
```cpp
// HOT: Frequently accessed, keep pointer-based
struct HotNode {
    int _nid;
    Type* _type;
    Node* _input0;  // Most common: first input
};

// COLD: Rarely accessed, store columnar
struct ColdNodeData {
    std::vector<Node*> extra_inputs;   // inputs[1..N]
    std::vector<Node*> outputs;
    metadata...
};
```

**Bandwidth win:** Hot path = compact, cold path = compressed.

### Strategy 2: Immutable Bulk Phases

**Key insight:** Graph construction vs optimization have different patterns.

**Construction phase (mutable):**
```cpp
// Use pointer-based nodes during construction
StandardNode* n = new StandardNode();
n->setInput(0, other);
// Fast, flexible, standard traversal
```

**Optimization phase (immutable):**
```cpp
// Convert to columnar for bulk queries
ColumnarGraph columnar = freezeGraph(pointer_graph);

// Run subsumption queries (read-only)
auto candidates = columnar.query()
    .whereType(ArithmeticOp)
    .whereDominated(loop)
    .extractIndices();

// Apply transformations in batch
for (size_t idx : candidates) {
    apply_peephole(idx);
}

// Convert back if modifications needed
pointer_graph = thawGraph(columnar);
```

**Bandwidth win:** Columnar only during read-heavy phases.

### Strategy 3: Tiered Columnar (Progressive Densification)

**Level 1: Pointer-based (standard nodes)**
```cpp
Node* regular_nodes[1000];  // First tier
```

**Level 2: Compressed indices**
```cpp
// Nodes referenced by index, not pointer
uint32_t node_indices[10000];  // 4 bytes vs 8 bytes
```

**Level 3: Bitmaps for properties**
```cpp
// Properties become bitsets
std::bitset<10000> is_arithmetic;
std::bitset<10000> is_loop_invariant;
std::bitset<10000> is_vectorizable;
// 1 bit per node per property
```

**Level 4: Compressed sparse row (CSR) for edges**
```cpp
struct CSREdges {
    std::vector<uint32_t> offsets;   // Start of each node's edges
    std::vector<uint32_t> targets;   // Target node indices
};
// Used in graph databases, extreme compression
```

**Bandwidth scaling:** Progressive compression as graph stabilizes.

---

## Click's Densification + Columnar

### Densification Goal: Minimize Unique Nodes

**Observation:** Fewer nodes = less memory bandwidth.

**Densification strategy:**
```cpp
// Hash consing: identical nodes merged
Node* getOrCreateConstant(int value) {
    auto it = constant_cache.find(value);
    if (it != constant_cache.end()) {
        return it->second;  // Reuse existing
    }
    auto* n = new ConstantNode(value);
    constant_cache[value] = n;
    return n;
}
```

**Columnar amplification:**
```cpp
// Densified nodes = more cache hits in columnar layout
// If 1000 constants deduplicated to 10:
// - Pointer-based: Still 1000 references, 1000 cache misses
// - Columnar: 10 unique entries, rest are indices (cache-friendly)
```

**Synergy:** Densification + columnar = extreme compression.

### Currying + Columnar

**Currying = type-based specialization**

Instead of:
```cpp
AddNode* add = new AddNode(lhs, rhs, type);  // Generic
```

Use:
```cpp
AddI32Node* add = new AddI32Node(lhs, rhs);  // Type-curried
```

**Columnar advantage:**
```cpp
// Separate columns per type
std::vector<AddI32Node> add_i32_nodes;  // Homogeneous
std::vector<AddF64Node> add_f64_nodes;  // Homogeneous

// Query becomes type-specialized scan
for (auto& add : add_i32_nodes) {
    // All nodes same layout, perfect cache streaming
}
```

**Bandwidth win:** Homogeneous arrays = perfect prefetching.

---

## When Columnar Wins vs Loses

### Wins (Low Drag)

| Operation | Bandwidth Reduction | Why |
|-----------|---------------------|-----|
| Type-based scan | 5-10x | Sequential access, single column |
| Bulk property extraction | 3-5x | Amortize traversal |
| Subsumption queries | 10-100x | Bitmap operations |
| Read-only analysis | 2-5x | No update overhead |
| SIMD-friendly ops | 4-8x | Packed data, vector instructions |

### Loses (High Drag)

| Operation | Bandwidth Increase | Why |
|-----------|-------------------|-----|
| Pointer chasing | 2-3x worse | Double indirection |
| Sparse updates | 10-100x worse | Column rewrites |
| Mixed-width access | 2-4x worse | Scattered reads |
| Incremental construction | 5-10x worse | Vector resizes |
| Random insertion/deletion | 50-100x worse | Array shifts |

---

## Optimal Strategy: Hybrid Architecture

### Phase 1: Construction (Pointer-Based)

```cpp
// Build graph with standard nodes
Graph* g = new Graph();
auto* n1 = g->createNode<AddNode>();
auto* n2 = g->createNode<MulNode>();
n2->setInput(0, n1);
```

**Why pointer-based:** Flexible, incremental, standard traversal.

### Phase 2: Freeze to Columnar

```cpp
// Convert to read-optimized columnar layout
ColumnarGraph* cg = g->freeze();

// Bulk analysis (bandwidth-efficient)
auto hoistable = cg->query()
    .whereLoopInvariant()
    .whereDominatedBy(loop)
    .extractIndices();
```

**Why columnar:** Read-heavy, bulk queries, bandwidth-critical.

### Phase 3: Thaw for Mutations

```cpp
// Convert back for modifications
Graph* g2 = cg->thaw();

// Apply transformations
for (size_t idx : hoistable) {
    g2->hoistNode(idx);
}
```

**Why pointer-based:** Mutations require flexibility.

### Phase 4: Final Freeze

```cpp
// Re-freeze for codegen
ColumnarGraph* final = g2->freeze();
emit_code(final);  // Bandwidth-efficient emission
```

---

## Subsumption Engine Integration

### Columnar-Friendly Subsumption

```cpp
class ColumnarSubsumptionEngine {
    // Bitmaps for fast filtering
    std::bitset<MAX_NODES> type_masks[NUM_TYPES];
    std::bitset<MAX_NODES> domination_masks[NUM_REGIONS];
    std::bitset<MAX_NODES> property_masks[NUM_PROPERTIES];

    auto query() {
        return QueryBuilder(this);
    }

    class QueryBuilder {
        std::bitset<MAX_NODES> _result = all_ones;

        QueryBuilder& whereType(TypeMask mask) {
            _result &= type_masks[mask];  // Single AND operation
            return *this;
        }

        QueryBuilder& whereDominated(RegionID region) {
            _result &= domination_masks[region];  // Single AND operation
            return *this;
        }

        std::vector<NodeIndex> extractIndices() {
            std::vector<NodeIndex> result;
            for (size_t i = 0; i < MAX_NODES; i++) {
                if (_result[i]) result.push_back(i);
            }
            return result;
        }
    };
};
```

**Bandwidth advantage:**
- Query = bitwise AND operations (SIMD)
- 512 nodes per cache line (1 bit each)
- Extreme compression for read-only queries

---

## Clinical Answer: Drag vs Bandwidth Trade-off

### When Columnar Adds Drag

1. **Random access patterns** (def-use chains) → 2-3x slower
2. **Sparse updates** (modify single node) → 10-100x slower
3. **Mixed-width access** (cross-column reads) → 2-4x slower
4. **Incremental construction** → 5-10x slower

### When Columnar Reduces Bandwidth

1. **Sequential scans** → 5-10x less bandwidth
2. **Bulk queries** → 10-100x less bandwidth (with bitmaps)
3. **Read-only analysis** → 3-5x less bandwidth
4. **Type-homogeneous operations** → 4-8x less bandwidth

### The Answer: Phase-Based Hybrid

**Construction:** Pointer-based (flexibility > bandwidth)
**Analysis:** Columnar (bandwidth > flexibility)
**Mutation:** Pointer-based (mutations require flexibility)
**Codegen:** Columnar (read-only, bandwidth-critical)

**No single layout wins everywhere.** Optimal strategy = phase-aware layout switching.

---

## Concrete Example: GCM Scheduling

### Without Columnar (Current Band 3)

```cpp
void GlobalCodeMotion::scheduleEarly() {
    for (Node* n : graph->nodes()) {        // Pointer chase
        if (!isPinned(n)) {                 // Load full node
            CFGNode* early = findEarliest(n); // More pointer chases
            n->setInput(0, early);          // Mutation
        }
    }
}
// Bandwidth: High (pointer chasing)
// Drag: Low (flexible mutations)
```

### With Hybrid Columnar

```cpp
void GlobalCodeMotion::scheduleEarly() {
    // Phase 1: Freeze to columnar for analysis
    auto* cg = graph->freeze();

    // Phase 2: Bulk query unpinned nodes (columnar win)
    auto unpinned = cg->queryUnpinned();  // Bitmap scan

    // Phase 3: Compute earliest positions (columnar win)
    std::vector<CFGNode*> earliest(unpinned.size());
    for (size_t i = 0; i < unpinned.size(); i++) {
        earliest[i] = cg->computeEarliest(unpinned[i]);  // Columnar read
    }

    // Phase 4: Thaw for mutations
    graph = cg->thaw();

    // Phase 5: Apply scheduling (pointer-based mutations)
    for (size_t i = 0; i < unpinned.size(); i++) {
        graph->node(unpinned[i])->setInput(0, earliest[i]);
    }
}
// Bandwidth: Lower (columnar analysis)
// Drag: Minimal (thaw only once)
```

**Net win:** Bandwidth reduced during read-heavy analysis, drag avoided by thawing for mutations.

---

## Conclusion

**Memory bandwidth conquered via:**
1. **Columnar layout for read-heavy phases** (queries, analysis)
2. **Pointer layout for write-heavy phases** (construction, mutation)
3. **Hybrid switching** based on operation mix
4. **Bitmap queries** for extreme compression (512x)
5. **Densification** to reduce unique nodes (amplifies columnar benefits)

**Drag factors avoided by:**
1. **Never using pure columnar** (always hybrid)
2. **Phase-based layout switching** (thaw when needed)
3. **Hot/cold splitting** (frequent fields stay pointer-based)
4. **Immutable optimization passes** (read-only = safe for columnar)

**The answer:** Columnar doesn't always add drag IF you switch layouts based on operation patterns. Pure columnar = disaster for mutations. Hybrid columnar = bandwidth win without drag penalty.
