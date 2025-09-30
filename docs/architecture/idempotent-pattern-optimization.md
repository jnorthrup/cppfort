# Idempotent Pattern Optimization: Recompute vs Store

**Key Insight:** Idempotent patterns with cheap recomputation don't need storage - recompute on-demand.

---

## Problem: Rainbow Table Storage Overhead

**Naive approach:** Store all pattern lowering results in rainbow table.

```cpp
// 1000 patterns × 7 opt levels = 7000 entries
// Each entry: NodeKind + OptLevel + LoweringStrategy + cost
// Memory: 7000 × ~100 bytes = 700KB minimum
```

**Problem:** Many patterns are idempotent (pure functions) with trivial recomputation cost.

**Example - Constant folding:**
```cpp
// AddNode(ConstantNode(2), ConstantNode(3))
// Stored result: ConstantNode(5)
// Storage cost: 64-128 bytes (node allocation)
// Recompute cost: 1 CPU cycle (integer add)

// Storage vs recompute: 64 bytes vs 1 cycle = wasteful
```

---

## Idempotent Pattern Categories

### 1. Trivial Recompute (< 10 cycles)

**Constant arithmetic:**
```cpp
AddNode(Const(a), Const(b)) → Const(a + b)  // 1 cycle
MulNode(Const(a), Const(b)) → Const(a * b)  // 3 cycles
```

**Storage policy:** RECOMPUTE_ON_DEMAND

**Rationale:** CPU cache miss (100-300 cycles) >> recomputation (1-10 cycles).

### 2. Moderate Recompute (10-100 cycles)

**Type inference:**
```cpp
InferType(node) → TypeLattice  // 20-50 cycles (lattice navigation)
```

**Storage policy:** STORE if accessed > 10x, else RECOMPUTE

**Rationale:** Amortize storage cost over repeated accesses.

### 3. Expensive Recompute (> 100 cycles)

**Complex constant folding:**
```cpp
PowNode(Const(2.0), Const(10.0)) → Const(1024.0)  // 200+ cycles (FPU pow)
```

**Storage policy:** STORE

**Rationale:** Recomputation cost > cache miss cost.

### 4. Non-Idempotent

**Patterns with side effects:**
```cpp
AllocNode() → different allocation each time (stateful)
```

**Storage policy:** STORE (cannot recompute)

---

## Implementation Strategy

### Rainbow Table Entry

```cpp
struct RainbowEntry {
    NodeKind pattern : 16;
    OptLevel opt_level : 8;
    LoweringStrategy lower : 8;
    uint32_t cost;

    // Idempotent optimization
    bool idempotent : 1;
    uint16_t recompute_cost : 15;  // Cycles to recompute
};
```

**Size:** 12 bytes per entry (packed).

### Dispatch Logic

```cpp
LoweringStrategy dispatch(NodeKind k, OptLevel o) {
    uint32_t hash = (k << 8) | o;
    const RainbowEntry& entry = rainbow_table[hash % TABLE_SIZE];

    // Idempotent optimization
    if (entry.idempotent && entry.recompute_cost < RECOMPUTE_THRESHOLD) {
        // Recompute on-demand (no storage access)
        return recompute_lowering(k, o);
    } else {
        // Fetch stored result
        return entry.lower;
    }
}
```

**RECOMPUTE_THRESHOLD:** Default 10 cycles (tunable).

### Recomputation Functions

```cpp
LoweringStrategy recompute_lowering(NodeKind k, OptLevel o) {
    switch (k) {
        case NodeKind::ADD_CONST:
            // Constant folding: always fold at -O1+
            return (o >= OptLevel::O1) ? Lowering::Fold : Lowering::Simple;

        case NodeKind::MUL_BY_POW2:
            // Multiply by power-of-2: shift at -O1+
            return (o >= OptLevel::O1) ? Lowering::Shift : Lowering::Simple;

        default:
            // Fallback to stored result
            return rainbow_table[hash % TABLE_SIZE].lower;
    }
}
```

**Inlining:** Compiler inlines these trivial switch statements (zero call overhead).

---

## Differential Tracking Integration

### Pattern Analysis

Differential tracking identifies idempotent patterns:

```python
def analyze_pattern_idempotence(pattern):
    """Determine if pattern is idempotent and recompute cost."""

    # Check purity (no side effects)
    if not pattern.is_pure_function():
        return {'idempotent': False, 'recompute_cost': float('inf')}

    # Estimate recompute cost
    cost = 0
    if pattern.is_constant_arithmetic():
        cost = 1  # Single ALU operation
    elif pattern.is_type_inference():
        cost = 30  # Type lattice navigation
    elif pattern.is_complex_math():
        cost = 200  # FPU operations
    else:
        cost = 50  # Default conservative estimate

    return {
        'idempotent': True,
        'recompute_cost': cost,
        'storage_policy': 'recompute' if cost < 10 else 'store'
    }
```

### Database Schema

```json
{
  "pattern_id": "add_const_fold",
  "source_pattern": "AddNode(ConstantNode, ConstantNode)",
  "idempotent": true,
  "recompute_cost": 1,
  "storage_policy": "recompute_on_demand",
  "differential": {
    "Og→O1": {"cost_reduction": 2, "eliminated": true},
    "O1→O2": {"cost_reduction": 0, "eliminated": false}
  }
}
```

**Storage policy determines:**
- Rainbow table inclusion (store) vs exclusion (recompute)
- Memory footprint reduction
- Cache pressure reduction

---

## Memory Savings Analysis

### Baseline (No Optimization)

- 1000 patterns
- 7 opt levels
- 100 bytes per entry
- **Total: 700KB**

### With COW (Handle/Body)

- 1000 pattern bodies × 100 bytes = 100KB (shared)
- 7000 handles × 8 bytes = 56KB
- **Total: 156KB (4.5x reduction)**

### With COW + Idempotent Recompute

**Assumptions:**
- 200/1000 patterns are idempotent with recompute_cost < 10
- These patterns don't need bodies stored

**Calculation:**
- 800 pattern bodies × 100 bytes = 80KB (stored)
- 200 patterns × 0 bytes = 0KB (recomputed)
- 7000 handles × 8 bytes = 56KB
- **Total: 136KB (5.1x reduction from baseline)**

**Additional benefit:** Reduced cache pressure (200 fewer body accesses).

---

## Performance Analysis

### Cache Miss Cost

**Modern CPU (2025):**
- L1 cache hit: 1-3 cycles
- L2 cache hit: 10-20 cycles
- L3 cache hit: 40-70 cycles
- DRAM miss: 100-300 cycles

### Recompute Cost vs Cache Miss

**Constant folding:**
```
Recompute: 1 cycle (ALU add)
Cache miss: 100-300 cycles (DRAM fetch)
Speedup: 100-300x (recompute wins)
```

**Type inference:**
```
Recompute: 30 cycles (lattice navigation)
Cache miss: 100-300 cycles (DRAM fetch)
Speedup: 3-10x (recompute still wins if entry not in cache)
```

**Complex math:**
```
Recompute: 200 cycles (FPU pow)
Cache miss: 100-300 cycles (DRAM fetch)
Speedup: 0.5-1.5x (store wins if accessed repeatedly)
```

### Access Pattern Sensitivity

**Single access patterns:**
- Recompute always wins (no cache benefit from storage)

**Repeated access patterns:**
- Recompute wins if cost < L3 miss (70 cycles)
- Store wins if cost > L3 miss and accessed > 2x

**Hot path patterns:**
- Recompute wins if cost < L1 miss (3 cycles)
- Store wins if cost > L1 miss (likely cached)

---

## Adaptive Threshold

### Runtime Profiling

```cpp
class AdaptiveRecomputeThreshold {
    uint64_t _recompute_count = 0;
    uint64_t _recompute_cycles = 0;
    uint64_t _cache_miss_count = 0;
    uint64_t _cache_miss_cycles = 0;

public:
    void on_recompute(uint64_t cycles) {
        _recompute_count++;
        _recompute_cycles += cycles;
    }

    void on_cache_miss(uint64_t cycles) {
        _cache_miss_count++;
        _cache_miss_cycles += cycles;
    }

    uint16_t optimal_threshold() const {
        // Average cache miss cost
        uint64_t avg_miss = _cache_miss_cycles / _cache_miss_count;

        // Threshold: recompute if cost < 50% of avg miss
        return static_cast<uint16_t>(avg_miss * 0.5);
    }
};
```

**Adaptive strategy:** Tune threshold based on actual hardware performance.

### Per-Pattern Profiling

```cpp
struct PatternProfile {
    uint64_t access_count;
    uint64_t recompute_cycles_total;

    bool should_store() const {
        // If accessed > 10x and avg recompute > 50 cycles, store
        uint64_t avg_recompute = recompute_cycles_total / access_count;
        return access_count > 10 && avg_recompute > 50;
    }
};
```

**Per-pattern policy:** Patterns can migrate from recompute → store based on runtime behavior.

---

## Integration with Stage2 Differential Tracking

### Marking Idempotent Patterns

```bash
# Extract idempotence from -Og IR
./scripts/analyze_idempotence.py \
    --ir-dump build/*.ir \
    --output idempotent_patterns.json

# Example output
{
  "add_const": {
    "idempotent": true,
    "pure": true,
    "recompute_cost": 1,
    "storage_policy": "recompute_on_demand"
  },
  "alloc_node": {
    "idempotent": false,
    "pure": false,
    "recompute_cost": "inf",
    "storage_policy": "store"
  }
}
```

### Rainbow Table Generation

```python
def generate_rainbow_table():
    idempotent_db = load_idempotent_patterns()

    for pattern in all_patterns:
        info = idempotent_db[pattern.id]

        for opt_level in opt_levels:
            cost = compute_cost(pattern, opt_level)
            lowering = select_lowering(pattern, opt_level, cost)

            entry = RainbowEntry(
                pattern=pattern.kind,
                opt_level=opt_level,
                lowering=lowering,
                cost=cost,
                idempotent=info['idempotent'],
                recompute_cost=info['recompute_cost']
            )

            # Only emit entry if storing (not recomputing)
            if info['storage_policy'] == 'store':
                emit_entry(entry)
            else:
                emit_recompute_marker(entry)
```

**Result:** Rainbow table contains only patterns requiring storage.

---

## Implementation Phases

### Phase 1: Mark Idempotent Patterns (Week 1)

- [ ] Analyze IR for pure functions
- [ ] Estimate recompute costs
- [ ] Tag patterns in differential database
- [ ] Generate idempotent_patterns.json

### Phase 2: Implement Recompute Functions (Week 2)

- [ ] Write recompute_lowering() dispatch
- [ ] Implement trivial recompute cases (const folding)
- [ ] Add profiling hooks
- [ ] Validate correctness vs stored results

### Phase 3: Adaptive Threshold (Week 3)

- [ ] Add runtime profiling infrastructure
- [ ] Measure cache miss costs
- [ ] Tune RECOMPUTE_THRESHOLD
- [ ] A/B test vs baseline

### Phase 4: Integration (Week 4)

- [ ] Update rainbow table generation
- [ ] Remove redundant storage
- [ ] Measure memory savings
- [ ] Performance validation

---

## Success Metrics

### Memory

- **Target:** 5x reduction in rainbow table memory footprint
- **Baseline:** 700KB (no optimization)
- **Expected:** 140KB (with COW + idempotent recompute)

### Performance

- **Target:** ≤ 5% regression on rainbow table dispatch latency
- **Baseline:** 10ns per dispatch (hash lookup + dereference)
- **Expected:** 10-12ns per dispatch (hash + conditional recompute)

### Coverage

- **Target:** 20-30% of patterns marked idempotent
- **Baseline:** 0% (all stored)
- **Expected:** ~200/1000 patterns (constant folding, type inference)

---

## Related Work

- **Rainbow Table** (`tblgen-optimization-rainbow-table.md`) - Storage structure
- **Differential Tracking** (`stage2-disasm-tblgen-differential.md`) - Pattern analysis
- **Handle/Body COW** (`tblgen-optimization-rainbow-table.md`) - Memory efficiency
- **Subsumption Engine** (`ADR-001`) - Pattern matching infrastructure

---

## Conclusion

**Idempotent patterns with cheap recomputation (< 10 cycles) don't need storage.**

**Strategy:**
1. Mark idempotent patterns during differential tracking
2. Estimate recompute cost (arithmetic, type inference, complex math)
3. Storage policy: recompute if cost < THRESHOLD (default 10 cycles)
4. Rainbow table contains only stored patterns
5. Dispatch checks idempotent flag, recomputes or fetches

**Benefits:**
- 5x memory reduction (700KB → 140KB)
- Reduced cache pressure (200 fewer body accesses)
- Negligible performance impact (recompute 1-10 cycles << cache miss 100-300 cycles)

**As long as patterns are idempotent, we don't need to store them unless recomputation is prohibitively expensive.**
