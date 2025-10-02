# Band Structure: Sea of Nodes Implementation Strategy

**Concept:** The Simple compiler chapters are grouped into "Bands" - coherent implementation phases that build on each other.

## Band Definitions

### Band 1: Foundation (Chapters 1-6)
**Theme:** Basic Sea of Nodes infrastructure

**Components:**
- StartNode, StopNode, ReturnNode
- ConstantNode (integer constants)
- Arithmetic operators (Add, Sub, Mul, Div, Minus)
- Scope and variable tracking
- Basic peephole optimizations
- Parser integration (direct graph construction, no AST)

**Deliverable:** Simple return statements and arithmetic compile to executable code.

**Git Tag:** band1 (if tagged)

---

### Band 2: Loops + Memory (Chapters 7-10)
**Theme:** Control flow complexity and memory model

**Components:**
- RegionNode (control flow merge)
- IfNode (conditional branches)
- LoopNode + PhiNode (iteration)
- Memory model (Load, Store, Proj)
- Struct types and field access
- NewNode (allocation)

**Deliverable:** Loops, conditionals, and heap allocations work correctly.

**Git Tag:** band2 (if tagged)

---

### Band 3: Global Code Motion (Chapter 11)
**Theme:** The critical scheduling transformation

**Components:**
- Early/late scheduling algorithms
- Dominator tree computation (idom, idepth)
- Loop depth analysis
- Anti-dependency insertion
- NeverNode (infinite loop handling)

**Status:** Implemented but with gaps (see QA gate FAIL assessment)

**Deliverable:** Unordered Sea of Nodes graph becomes scheduled executable code.

**Git Commit:** c8ad3b1 "Band 3: Global Code Motion (Chapter 11)"

---

### Band 4: Type System Expansion (Chapters 12-15)
**Theme:** Production-ready type system

**Components:**

**Chapter 12 - Floating Point:**
- TypeFloat (f32, f64)
- Float arithmetic nodes
- Float comparisons
- I2F, F2I conversions

**Chapter 13 - References:**
- TypePointer (nullable vs non-nullable)
- Null safety analysis
- Null check elimination

**Chapter 14 - Narrow Types:**
- TypeNarrow (i8, i16, i32, i64, u8, u16, u32)
- Widening/narrowing conversions
- Range analysis
- Overflow checking

**Chapter 15 - Arrays:**
- TypeArray (fixed and dynamic)
- NewArray, ALoad, AStore, ArrayLength
- Bounds checking (static and dynamic)

**Status:** Implemented (commit f49ac4b)

**Deliverable:** Full type system supporting real-world C/C++/CPP2 code.

**Git Commit:** f49ac4b "Band 4: Type System Expansion (Chapters 12-15)"

---

### Band 5: Constructors & Mutability (Chapters 16-17)
**Theme:** Struct constructors, final fields, and mutability semantics.

**Components:**
- **Chapter 16: Constructors and Final Fields**
  - `TypeStruct` with field metadata (finality, defaults)
  - `NewNode` enhancements for constructor initializers
  - Initialization validation for final and non-nullable fields
- **Chapter 17: Syntax Sugar - Mutability and Type Inference**
  - Field and pointer mutability qualifiers (`var`, `val`, `!`)
  - Deep immutability rules
  - Foundation for type inference (`glb`)

**Status:** ✅ Complete. See [review](../qa/band5_ch16_ch17_review.md).

---

### Band 6+: Advanced Optimizations (Chapters 18-24)
**Theme:** Production-strength optimization passes

**Likely Components:**
- **Escape analysis** (arena allocation, stack promotion)
- **Borrow checking** (Rust-like safety via graph dominance)
- **Arena allocators** ("one-way drones" - bump allocation, bulk deallocation)
- Inlining
- Loop optimizations (fusion, unrolling, invariant code motion)
- Vectorization (SIMD)
- Dead code elimination
- Alias analysis refinement
- Code generation
- Target-specific lowering

**Status:** Not yet implemented

**Deliverable:** Optimizing compiler competitive with Clang/GCC -O2 with Rust-level memory safety.

**See:** [Borrow Checking & Arena Allocators](borrow-checking-arenas.md) for memory safety design.

---

## Band Philosophy

### Why Bands?

1. **Coherent Milestones** - Each band delivers meaningful functionality
2. **Testable Units** - Tests written per band, not per chapter
3. **Incremental Complexity** - Each band builds on stable foundation
4. **Parallel Development** - Multiple people can work on different bands
5. **Risk Management** - Can ship Band N without Band N+1

### Band Dependencies

```
Band 1 (Foundation)
   ↓
Band 2 (Loops + Memory)
   ↓
Band 3 (Scheduling) ← CRITICAL CONVERGENCE POINT
   ↓
Band 4 (Types)
   ↓
Band 5 (Constructors & Mutability)
   ↓
Band 6+ (Advanced Opts)
```

**Band 3 is the critical chokepoint:** All optimization passes must respect scheduling.

### Subsumption Engine Relationship

**CRITICAL:** Band boundaries ≠ Subsumption boundaries.

The subsumption engine is **cross-band infrastructure** that queries concepts from ALL bands:

```cpp
// Single query spans Band 2, 3, 4 concepts
auto optimizable = engine.query()
    .whereCFG(dominatedBy(loop))           // Band 2 concept
    .whereSchedule(hoistable())            // Band 3 concept
    .whereType(subsumes<NumericType>());   // Band 4 concept
```

**Subsumption boundaries** are dynamic partitions based on optimization concerns (type, memory, CFG, etc.), NOT implementation phases.

**Bands** are temporal milestones for implementing features sequentially.

See [Subsumption Boundaries vs Bands](subsumption-boundaries-vs-bands.md) for detailed explanation.

## Implementation Status

| Band | Chapters | Status | Git Commit | Notes |
|------|----------|--------|------------|-------|
| 1 | 1-6 | ✓ Complete | (multiple) | Foundation working |
| 2 | 7-10 | ✓ Complete | d914415+ | Loops + memory working |
| 3 | 11 | ⚠️ Partial | c8ad3b1 | GCM implemented but has gaps |
| 4 | 12-15 | ✓ Complete | f49ac4b | Type system expansion done |
| 5  | 16-17 | ✅ Complete | f0b9243 | See [review](../qa/band5_ch16_ch17_review.md) |
| 6+ | 18-24 | ❌ Not started | - | Advanced opts pending |

## Current Focus: Band 3 Quality + Subsumption Foundation

**Immediate priorities:**
1. Fix Band 3 gaps identified in QA assessment
2. Design subsumption engine API
3. Integrate subsumption with Band 3 GCM
4. Prepare subsumption infrastructure for Band 5+ optimization passes

## Testing Strategy Per Band

### Band 1-2
- Chapter-specific tests (test_chapter1.cpp, test_chapter2.cpp, etc.)
- Working and passing

### Band 3
- test_band3.cpp currently theatrical (non-compilable)
- Needs real tests for scheduling correctness

### Band 4
- No dedicated band4 tests yet
- Should test type conversions, null safety, bounds checking

### Band 5
- Chapter-specific tests (`test_chapter16.cpp`, `test_chapter17.cpp`)
- All passing.

### Band 6+
- Will need comprehensive optimization benchmarks
- Performance regression testing
- Comparison with Clang/GCC output

## Git Workflow

### Recommended Commit Strategy

```bash
# Feature work
git commit -m "band3: Implement anti-dependency detection"

# Band completion
git commit -m "Band 3: Global Code Motion (Chapter 11)"
git tag -a band3 -m "Complete GCM implementation"

# Cross-band infrastructure
git commit -m "subsumption: Add hash-based primary index (for Band 3+)"
```

### Branch Strategy

- `master` - Stable, tested bands
- `feature/band-N` - Work-in-progress for Band N
- `feature/subsumption-engine` - Cross-band infrastructure

## Success Metrics Per Band

**Band 1:** Return statements and arithmetic work
**Band 2:** Loops and memory allocations work
**Band 3:** Scheduling produces correct execution order
**Band 4:** All type conversions preserve semantics
**Band 5:** Correct constructor and mutability semantics enforced.
**Band 6+:** Performance competitive with commercial compilers

## Related Documents

- [Band 3 GCM Documentation](../band3_gcm.md)
- [Band 4 Type System Documentation](../band4_types.md)
- [Subsumption Engine Architecture](subsumption-engine.md)
- [Subsumption Densification Gains](subsumption-densification-gains.md)
- [QA Gate: Band 3 Assessment](../qa/gates/band3-gcm-c8ad3b1.yml)
