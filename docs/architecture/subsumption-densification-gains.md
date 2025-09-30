# Subsumption-Based Densification: Optimization Gains Across Chapters

**Architect:** Winston
**Date:** 2025-09-30
**Context:** Click's densification philosophy + Subsumption engine capabilities

## The Core Insight: Coinserter for Each Interest

Your metaphor is precise: the subsumption engine acts as a **coinserter for each optimization interest**, partitioning the graph by concern and applying specialized rules to each partition simultaneously.

### Traditional Approach (Linear Passes)
```
Full Graph → Type Pass → Full Graph → Memory Pass → Full Graph → Loop Pass
   O(n)         O(n)        O(n)          O(n)         O(n)       O(n)
= 6n traversals
```

### Subsumption Approach (Orthogonal Consolidation)
```
Full Graph → Partition by Interest → Apply Rules in Parallel
   O(n)            O(n)                    O(p)
= n + p where p = operations per partition << n
```

## Click's Densification Philosophy

### Core Principles (from Click & Palsberg)

1. **Maximize Densification** - Aggressively merge equivalent nodes
2. **Minimize Memory Bandwidth** - Reduce loads/stores greedily
3. **Currying Tendency** - Specialize operations for specific types/values
4. **Global Value Numbering** - Hash consing everywhere possible
5. **Eager Peepholing** - Apply optimizations during construction

### How Subsumption Amplifies This

The subsumption engine enables **bulk densification** across type hierarchies:

```cpp
// Traditional: Check each Add node individually
for (Node* n : graph) {
    if (AddNode* add = dynamic_cast<AddNode*>(n)) {
        if (add->lhs()->isConstant() && add->rhs()->isConstant()) {
            fold(add);
        }
    }
}

// Subsumption: Partition once, fold all
auto foldable = engine.query()
    .whereType(subsumes<ArithmeticOp>())  // All arithmetic at once
    .whereDataFlow(allInputsConstant())
    .projectToConstants();
```

## Gains Across Upcoming Chapters

### Band 4: Type System (Chapters 12-15)

#### Chapter 12: Floating Point

**Subsumption Opportunity:**
```cpp
// Query all numeric operations that can widen/narrow together
auto numericOps = engine.query()
    .whereType(subsumes<NumericType>())  // int + float hierarchy
    .groupBy(compatiblePrecision())
    .projectConversions();

// Bulk cast elimination
numericOps.forEach([](auto& group) {
    eliminateRedundantCasts(group);  // i32→f64→i32 = identity
});
```

**Densification Gains:**
- **Cast chain folding** - Eliminate intermediate conversions across subgraphs
- **Precision specialization** - Curry f32 vs f64 paths separately
- **SIMD opportunities** - Identify vectorizable float ops by compatible precision

#### Chapter 13: Reference Types

**Subsumption Opportunity:**
```cpp
// Nullable analysis via type subsumption
auto nullChecks = engine.query()
    .whereType(subsumes<ReferenceType>())
    .whereNullability(nullable())
    .groupByControlFlow(dominatedBy(nullCheck));

// Collective null check elimination
auto proven = nullChecks.whereControlFlow(nonNullProven());
proven.eliminateDominatedChecks();  // Kill redundant checks in bulk
```

**Densification Gains:**
- **Null check hoisting** - Move checks to dominator, remove all dominated
- **Reference specialization** - Curry nullable vs non-nullable paths
- **Escape analysis grouping** - Partition allocations by escape scope

#### Chapter 14: Narrow Types

**Subsumption Opportunity:**
```cpp
// Width-based partitioning for range analysis
auto narrowOps = engine.query()
    .whereType(subsumes<IntegerType>())
    .groupBy(bitWidth())
    .projectRanges();

// Collective overflow elimination
narrowOps.forEach([](auto& widthGroup) {
    if (rangeProvenInBounds(widthGroup)) {
        widthGroup.eliminateOverflowChecks();
    }
});
```

**Densification Gains:**
- **Range-based DCE** - Eliminate bounds checks for entire regions
- **Width specialization** - Curry i8/i16/i32/i64 operations separately
- **Pack opportunities** - Group narrow ops for sub-word parallelism

#### Chapter 15: Arrays

**Subsumption Opportunity:**
```cpp
// Array operations partitioned by element type
auto arrayOps = engine.query()
    .whereType(subsumes<ArrayType>())
    .groupBy(elementType())
    .projectMemoryAccess();

// Collective bounds check elimination
arrayOps.forEach([](auto& typeGroup) {
    if (loopIndexProvenInBounds(typeGroup)) {
        typeGroup.eliminateBoundsChecks();
        typeGroup.hoistLengthAccess();
    }
});
```

**Densification Gains:**
- **Bounds check consolidation** - One check per loop, not per access
- **Memory access grouping** - Partition by alias class for disambiguation
- **Type-specialized loads** - Curry different element types separately

### Band 5+: Advanced Optimizations (Chapters 16-24)

#### Escape Analysis (Likely Chapter 16-17)

**Subsumption Opportunity:**
```cpp
// Partition allocations by escape scope
auto allocs = engine.query()
    .whereType<NewNode>()
    .groupBy(escapeScope())
    .projectLifetime();

// Stack allocate entire escope scopes
auto stackable = allocs.whereEscape(doesNotEscape());
stackable.lowerToStack();  // Bulk optimization
```

**Currying Advantage:**
- Specialize heap vs stack paths completely
- Eliminate GC overhead for proven-local allocations

#### Loop Optimization (Chapter 18-20)

**Subsumption Opportunity:**
```cpp
// Multi-loop analysis via nesting subsumption
auto loops = engine.query()
    .whereCFG(subsumes<LoopNode>())
    .groupBy(nestingDepth())
    .projectIterationSpace();

// Collective loop fusion
loops.whereFusible().forEach([](auto& fusibleSet) {
    fuseLoops(fusibleSet);  // Merge iteration spaces
});
```

**Densification Gains:**
- **Loop fusion** - Merge compatible loops to reduce memory traffic
- **Invariant hoisting** - Hoist across multiple loops simultaneously
- **Induction variable sharing** - Curry loop counters across fused loops

#### Vectorization (Chapter 21-22)

**Subsumption Opportunity:**
```cpp
// Vectorizable operations via type + control subsumption
auto vectorizable = engine.query()
    .whereType(subsumes<SIMDCompatible>())
    .whereControlFlow(straightLine())
    .whereDataFlow(noAliasing())
    .groupBy(vectorWidth());

// Bulk SIMD lowering
vectorizable.forEach([](auto& group) {
    lowerToMLIRVector(group);  // TableGen pattern match
});
```

**Densification Gains:**
- **SIMD packing** - Group 4/8/16 scalar ops into vector ops
- **Memory bandwidth reduction** - Vector loads instead of scalar
- **Specialized vector dialects** - Curry AVX vs NEON vs SVE

## The "Coinserter for Each Interest" Pattern

### Interest Categories

Each optimization concern gets its own "coinserter" (partition):

1. **Type Interest** - Group by type hierarchy position
2. **Memory Interest** - Partition by alias class
3. **Control Interest** - Group by domination/loop nesting
4. **Arithmetic Interest** - Partition by operation class
5. **MLIR Dialect Interest** - Group by target dialect compatibility

### Collective Consolidation Algorithm

```cpp
template<typename Interest>
void consolidate(Interest interest) {
    // 1. Partition graph by interest
    auto partitions = engine.query()
        .groupBy(interest.criterion());

    // 2. Apply interest-specific subsumption rules
    partitions.forEach([&](auto& partition) {
        auto subsumable = partition.whereSubsumed(interest.rules());
        subsumable.consolidate();  // Bulk merge
    });

    // 3. Project to target representation
    return partitions.project(interest.target());
}
```

### Example: Regions as Type-Based Coinserters

```cpp
// RegionNode with type-based consolidation
class RegionNode : public CFGNode {
public:
    // Coinserter for arithmetic interest
    auto arithmeticOps() {
        return subsumption_engine.query()
            .whereCFG(dominatedBy(this))
            .whereType(subsumes<ArithmeticOp>())
            .groupBy(resultType());
    }

    // Coinserter for memory interest
    auto memoryOps() {
        return subsumption_engine.query()
            .whereCFG(dominatedBy(this))
            .whereType(subsumes<MemoryOp>())
            .groupBy(aliasClass());
    }

    // Collective optimization
    void optimizeRegion() {
        arithmeticOps().forEach(constantFold);
        memoryOps().forEach(loadStoreElimination);
    }
};
```

## Click's Currying Tendency + Subsumption

### Currying via Type Specialization

Traditional currying (functional programming):
```haskell
add x y = x + y
add5 = add 5  -- Partially applied
```

Click's currying (type-based specialization):
```cpp
// Generic operation
Node* add(Node* lhs, Node* rhs);

// Type-curried variants (specialized)
auto addInt32 = engine.query()
    .whereType<AddNode>()
    .whereOperandType(TypeInteger::INT32)
    .specialize([](auto* add) {
        return new AddI32Node(add->lhs(), add->rhs());
    });

auto addFloat64 = engine.query()
    .whereType<AddNode>()
    .whereOperandType(TypeFloat::F64)
    .specialize([](auto* add) {
        return new AddF64Node(add->lhs(), add->rhs());
    });
```

### Why This Matters

**Memory Bandwidth Reduction:**
- Specialized ops = smaller IR footprint
- Type dispatch eliminated (curried away)
- Better instruction cache locality

**Densification:**
- More aggressive constant folding per type
- Type-specific algebraic identities
- Specialized peephole rules

## Greedy RAM Bandwidth Minimization

### Click's Greedy Strategy

1. **Hoist loads aggressively** - Move to earliest legal position
2. **Sink stores lazily** - Move to latest legal position
3. **Merge memory operations** - Consolidate loads/stores to same location
4. **Specialize memory paths** - Curry by alias class

### Subsumption Amplification

```cpp
// Greedy load hoisting via dominator subsumption
auto loads = engine.query()
    .whereType<LoadNode>()
    .whereMemory(noAliasing())
    .groupBy(dominatorTree());

loads.forEach([](auto& domGroup) {
    auto earliest = domGroup.findEarliestLegalPosition();
    domGroup.hoistAll(earliest);  // Bulk hoist
});

// Greedy store sinking via use subsumption
auto stores = engine.query()
    .whereType<StoreNode>()
    .whereMemory(noAliasing())
    .groupBy(leastCommonAncestor(uses()));

stores.forEach([](auto& lcaGroup) {
    auto latest = lcaGroup.findLatestLegalPosition();
    lcaGroup.sinkAll(latest);  // Bulk sink
});
```

**Result:** Minimize live ranges, maximize register reuse, reduce memory traffic.

## Integration with MLIR N-Way Conversions

### Problem: Multiple Target Dialects

Sea of Nodes must lower to:
- **Arithmetic dialect** - arith.addi, arith.mulf
- **Memory dialect** - memref.load, memref.store
- **Control dialect** - cf.br, cf.cond_br
- **Vector dialect** - vector.add, vector.fma
- **GPU dialect** - gpu.launch, gpu.barrier

### Subsumption Solution: N-Way Projection

```cpp
// Partition graph by target dialect compatibility
auto dialects = engine.query()
    .projectToMLIR()
    .groupBy([](Node* n) {
        return {
            .arithmetic = isArithmetic(n),
            .memory = isMemory(n),
            .control = isControl(n),
            .vectorizable = isVectorizable(n),
            .gpuable = isGPUCompatible(n)
        };
    });

// Each partition gets specialized lowering
dialects.arithmetic().lowerToArithDialect();
dialects.memory().lowerToMemRefDialect();
dialects.control().lowerToCFDialect();

// N-way intersections get hybrid lowering
auto vectorMemory = dialects.vectorizable() ∩ dialects.memory();
vectorMemory.lowerToVectorMemRefOps();  // Specialized pattern
```

## Concrete Example: Entire Optimization Pipeline

```cpp
void optimizeGraphWithSubsumption(Graph* graph) {
    SubsumptionEngine engine(graph);

    // 1. Type-based coinserter
    auto byType = engine.partitionByType();
    byType.constants().fold();           // Curry constants
    byType.arithmetic().simplify();      // Curry arithmetic
    byType.memory().disambiguate();      // Curry memory ops

    // 2. Control-based coinserter
    auto byCFG = engine.partitionByCFG();
    byCFG.loops().hoistInvariants();     // Curry loop bodies
    byCFG.regions().mergePhis();         // Curry merge points

    // 3. Data-flow coinserter
    auto byDefUse = engine.partitionByDataFlow();
    byDefUse.chains().eliminateRedundancy();

    // 4. MLIR projection
    auto mlir = engine.projectToMLIR();
    mlir.byDialect().applyPatterns();    // Curry per dialect

    // All done in O(n + p) instead of O(n * passes)
}
```

## Summary: The Subsumption Advantage

### What We Gain

1. **Bulk Consolidation** - Apply rules to entire partitions, not individual nodes
2. **Orthogonal Optimization** - Multiple concerns addressed simultaneously
3. **Type-Based Currying** - Specialize operations by type hierarchy
4. **Memory Bandwidth Greed** - Aggressive load/store consolidation
5. **N-Way Projections** - Lower to multiple target dialects efficiently
6. **Coinserter Pattern** - Each optimization interest gets dedicated partition

### Click's Philosophy + Subsumption = Synergy

- **Densification:** Subsumption enables bulk node merging
- **Currying:** Type hierarchies enable aggressive specialization
- **Bandwidth Minimization:** Memory operation partitioning enables greedy consolidation
- **Peepholing:** Pattern matching across partitions finds more opportunities

### The Core Formula

```
Orthogonal Hierarchical Consolidation =
    Partition by Interest (coinserter) +
    Apply Subsumption Rules (hierarchy) +
    Project to Target (n-way conversion)

Result: O(n) traversal, O(p) optimization per partition
```

This is the architectural foundation for scaling Click's aggressive optimization philosophy across all upcoming chapters.
