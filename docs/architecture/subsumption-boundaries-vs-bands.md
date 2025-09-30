# Subsumption Boundaries vs Band Boundaries

**CRITICAL DISTINCTION:** Band boundaries (implementation phases) are NOT subsumption boundaries (query partitions).

## The Confusion

**WRONG ASSUMPTION:**
- Band 1 = one subsumption domain
- Band 2 = another subsumption domain
- Band 3 = another subsumption domain

**REALITY:**
- Bands = implementation milestones (WHEN features are built)
- Subsumption boundaries = query partitions (HOW graph is queried)

These are **orthogonal concerns**.

---

## Band Boundaries (Implementation Organization)

**Purpose:** Pedagogical/organizational structure for implementing Sea of Nodes features.

**Boundaries defined by:** Chapter groupings from Simple compiler book.

**Examples:**
- **Band 1** (Chapters 1-6): Basic nodes, arithmetic, scope
- **Band 2** (Chapters 7-10): Loops, memory model, structs
- **Band 3** (Chapter 11): Scheduling (GCM)
- **Band 4** (Chapters 12-15): Type system expansion

**Nature:** These are **temporal** boundaries (what gets implemented when).

**Characteristics:**
- Sequential implementation order
- Testable milestones
- Additive (each band builds on previous)
- Fixed by book structure

---

## Subsumption Boundaries (Query Partitions)

**Purpose:** Operational structure for querying/optimizing the graph.

**Boundaries defined by:** Optimization concerns and node properties.

**Examples:**
- **Type boundary:** All nodes of compatible types
- **Memory boundary:** All nodes with same alias class
- **Control boundary:** All nodes dominated by specific CFG node
- **Lifetime boundary:** All nodes with same lifetime region
- **Dialect boundary:** All nodes lowerable to specific MLIR dialect

**Nature:** These are **logical** boundaries (how graph is partitioned for queries).

**Characteristics:**
- Orthogonal to implementation order
- Dynamic (computed at query time)
- Overlapping (node can be in multiple partitions)
- Defined by subsumption engine queries

---

## Concrete Example: Different Boundaries

### Band Boundary Example

**Band 2 implements:**
- LoopNode
- PhiNode
- Load/Store nodes
- Memory projections

**Band 3 implements:**
- Early/late scheduling
- Dominator tree
- Anti-dependencies

These are implementation phases - you build Band 2 features, then Band 3 features.

### Subsumption Boundary Example

**Memory subsumption query:**
```cpp
// Query spans nodes from MULTIPLE bands
auto memoryOps = engine.query()
    .whereType(subsumes<MemoryOp>())  // Band 2+ nodes
    .whereMemory(aliasClass(3))
    .groupBy(dominatingLoop());       // Band 3 concept
```

**Result:** Partition contains:
- Load nodes (Band 2)
- Store nodes (Band 2)
- Scheduled by GCM (Band 3)
- Typed by type system (Band 4)

**All from different bands, but in SAME subsumption partition.**

---

## Why The Distinction Matters

### 1. Subsumption Spans Bands

The subsumption engine queries **across** band boundaries:

```cpp
// This query touches concepts from Bands 2, 3, and 4
auto optimizable = engine.query()
    .whereCFG(dominatedBy(loop))       // Band 2 concept
    .whereSchedule(hoistable())        // Band 3 concept
    .whereType(subsumes<NumericType>()) // Band 4 concept
    .projectOptimizations();
```

**Subsumption doesn't care about implementation order.**

### 2. Bands Don't Define Query Boundaries

Band structure is for **humans** (implementation organization).
Subsumption structure is for **compiler** (optimization queries).

**Example:**
- Band 3 adds scheduling
- But scheduling queries can examine Band 2 nodes (loops)
- And apply Band 4 type information
- All in single subsumption query

### 3. Multiple Subsumption Boundaries Per Band

Band 4 (type system) alone enables multiple subsumption boundaries:

```cpp
// Boundary 1: Integer types
auto intOps = engine.whereType(subsumes<IntegerType>());

// Boundary 2: Float types
auto floatOps = engine.whereType(subsumes<FloatType>());

// Boundary 3: Reference types
auto refOps = engine.whereType(subsumes<ReferenceType>());

// Boundary 4: Array types
auto arrayOps = engine.whereType(subsumes<ArrayType>());
```

**One band, four+ subsumption boundaries.**

### 4. Subsumption Boundaries Are Dynamic

```cpp
// Same nodes, different boundaries depending on query
auto loop = findLoop();

// Boundary 1: By domination
auto dominated = engine.whereCFG(dominatedBy(loop));

// Boundary 2: By loop depth
auto nested = engine.whereLoopDepth(greaterThan(loop->depth()));

// Boundary 3: By invariance
auto invariants = engine.whereDataFlow(invariantTo(loop));
```

**Same graph, different partitions, depending on optimization concern.**

---

## Correct Mental Model

### Band Boundaries (Temporal)

```
Time →
Band 1 | Band 2 | Band 3 | Band 4 | Band 5+
[impl] | [impl] | [impl] | [impl] | [impl]
```

Linear, sequential, fixed.

### Subsumption Boundaries (Logical)

```
Graph Nodes
    ↓ Query 1: Type-based partition
    [IntOps] [FloatOps] [RefOps] [ArrayOps]
    ↓ Query 2: CFG-based partition
    [InLoop] [OutsideLoop] [Dominated] [NotDominated]
    ↓ Query 3: Memory-based partition
    [Alias0] [Alias1] [Alias2] [NoAlias]
    ↓ Query 4: Lifetime-based partition
    [Region1] [Region2] [Region3] [Escaped]
```

Multidimensional, dynamic, query-dependent.

---

## Subsumption Engine Relationship to Bands

### The Subsumption Engine Is Cross-Band Infrastructure

```
Band 1 ──┐
Band 2 ──┼─→ Subsumption Engine ──→ Query Partitions
Band 3 ──┤                            (dynamic boundaries)
Band 4 ──┤
Band 5+ ─┘
```

**The engine queries ALL implemented bands simultaneously.**

### Each Band Adds Query Dimensions

- **Band 1:** Node types, constants
- **Band 2:** Memory operations, control flow
- **Band 3:** Dominance, scheduling, loop depth
- **Band 4:** Type hierarchy, nullability, ranges
- **Band 5+:** Escape, inlining, vectorizability

**More bands = richer query vocabulary, but boundaries still dynamic.**

---

## Examples: Queries That Span Bands

### Example 1: Loop-Invariant Code Motion

**Uses concepts from multiple bands:**

```cpp
auto hoistable = engine.query()
    .whereCFG(inLoop(loop))              // Band 2 concept
    .whereSchedule(earliestAt(preheader)) // Band 3 concept
    .whereDataFlow(invariantInputs())    // Band 2 concept
    .whereType(isNumeric())              // Band 4 concept
    .projectToHoist();
```

**Subsumption boundary:** "Hoistable operations"
**Spans:** Bands 2, 3, 4

### Example 2: Vectorization Candidates

**Uses concepts from multiple bands:**

```cpp
auto vectorizable = engine.query()
    .whereCFG(straightLine())            // Band 2 concept
    .whereSchedule(scheduledSequentially()) // Band 3 concept
    .whereType(compatibleWidth(4))       // Band 4 concept
    .whereMemory(noAliasing())           // Band 2 concept
    .whereEscape(local())                // Band 5 concept
    .projectToSIMD();
```

**Subsumption boundary:** "Vectorizable operations"
**Spans:** Bands 2, 3, 4, 5

### Example 3: Arena Allocation Candidates

**Uses concepts from multiple bands:**

```cpp
auto arenaAllocatable = engine.query()
    .whereType<NewNode>()                // Band 2 concept
    .whereEscape(doesNotEscape())        // Band 5 concept
    .whereCFG(dominatedBy(region))       // Band 3 concept
    .whereLifetime(boundedBy(region))    // Band 5 concept
    .projectToArena();
```

**Subsumption boundary:** "Arena-allocatable objects"
**Spans:** Bands 2, 3, 5

---

## Architectural Implications

### 1. Don't Design Subsumption Per Band

**Wrong:**
```cpp
class Band2SubsumptionEngine { /* queries for Band 2 */ };
class Band3SubsumptionEngine { /* queries for Band 3 */ };
```

**Right:**
```cpp
class SubsumptionEngine {
    // Single engine, queries ALL bands
    QueryBuilder query();
};
```

### 2. Subsumption API Doesn't Care About Bands

**The query API is concern-based, not band-based:**

```cpp
// Good: Concern-based queries
engine.whereType(...)
engine.whereCFG(...)
engine.whereMemory(...)
engine.whereEscape(...)

// Bad: Band-based queries (DON'T DO THIS)
engine.whereBand2(...)  // ❌
engine.whereBand3(...)  // ❌
```

### 3. Implementation Order ≠ Query Order

**You implement features sequentially (bands), but query them arbitrarily:**

- Implement: Band 1 → Band 2 → Band 3 → Band 4
- Query: Any combination of concepts from any bands

**This is the power of the subsumption engine.**

---

## Summary

| Aspect | Band Boundaries | Subsumption Boundaries |
|--------|----------------|------------------------|
| **Purpose** | Implementation organization | Query partitioning |
| **Nature** | Temporal (when) | Logical (how) |
| **Structure** | Linear, sequential | Multidimensional, dynamic |
| **Defined by** | Book chapters | Optimization concerns |
| **Boundaries** | Fixed (1, 2, 3, 4, 5+) | Dynamic (query-dependent) |
| **Span** | One phase at a time | Span multiple bands |
| **For** | Human developers | Compiler queries |

---

## Corrected Understanding

**Bands** = Milestone structure for implementing Sea of Nodes features
**Subsumption** = Query mechanism that operates across ALL implemented features

**No 1:1 relationship between band boundaries and subsumption boundaries.**

The subsumption engine is **cross-band infrastructure** that enables querying the entire graph using concepts from whatever bands have been implemented, with boundaries determined by optimization concerns, not implementation order.
