# Band Structure: Dovetailing with Real-World Frontend IR Concepts

**Strategic Design:** The band structure wasn't arbitrary - each band was chosen to align with proven frontend IR abstractions used in production compilers.

## The Core Alignment

Sea of Nodes isn't an ivory tower design. Click's architecture deliberately dovetails with battle-tested IR concepts from:
- **LLVM** (SSA, dominator trees, memory model)
- **MLIR** (dialects, pattern matching, multi-level lowering)
- **JVM HotSpot** (Click's original implementation)
- **V8/SpiderMonkey** (modern JIT compilers)
- **GCC** (gimple, RTL, dominator-based opts)

### Why This Matters

**Interoperability:** Each band maps cleanly to existing IR concepts, enabling:
- Lowering to LLVM IR
- Conversion to MLIR dialects
- Integration with existing toolchains
- Reuse of proven optimization algorithms

## Band-by-Band IR Alignment

### Band 1: Foundation → SSA Form

**Sea of Nodes Concept:**
- Nodes with unique IDs
- Explicit def-use edges
- Value numbering

**Real-World IR Equivalent:**
```llvm
; LLVM SSA
%1 = add i32 %a, %b
%2 = mul i32 %1, %c
ret i32 %2
```

**Dovetail Points:**
- Sea of Nodes nodes = LLVM `%N` values
- Bidirectional edges = Use-Def chains
- Constant folding = LLVM constant propagation

**Why This Band First:**
SSA is the universal foundation. Get this right, and everything else builds cleanly.

---

### Band 2: Loops + Memory → CFG + Memory SSA

**Sea of Nodes Concept:**
- RegionNode (control merge)
- LoopNode + PhiNode
- Memory state explicit (Load, Store, Proj)

**Real-World IR Equivalent:**
```llvm
; LLVM CFG + Memory SSA
entry:
  %mem0 = call token @llvm.mem.ssa.token()
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %mem1 = phi token [ %mem0, %entry ], [ %mem2, %loop ]
  %mem2 = store i32 %i, ptr %array, align 4, !mem.ssa %mem1
  %i.next = add i32 %i, 1
  %cmp = icmp slt i32 %i.next, 100
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
```

**Dovetail Points:**
- RegionNode = LLVM basic block with multiple predecessors
- PhiNode = LLVM `phi` instruction
- Memory Proj = LLVM Memory SSA tokens
- LoopNode = Natural loop detection (same algorithm)

**Why This Band Second:**
CFG + memory model is the second universal abstraction. Loops are where optimization complexity explodes.

---

### Band 3: GCM → Instruction Scheduling + Dominator Trees

**Sea of Nodes Concept:**
- Early/late scheduling
- Dominator tree (idom, idepth)
- Anti-dependencies

**Real-World IR Equivalent:**
```
; LLVM Scheduling Concepts
- MachineScheduler (pre-RA and post-RA)
- DominatorTree analysis
- Anti-dependencies in ScheduleDAG
```

**Dovetail Points:**
- Early schedule = ASAP (as soon as possible) scheduling
- Late schedule = ALAP (as late as possible) scheduling
- idom() = LLVM DominatorTree::getIDom()
- Anti-deps = LLVM ScheduleDAG::addAntiDependency()

**MLIR Alignment:**
```mlir
// MLIR scheduling attributes
#scheduling = #llvm.scheduling<
  strategy = "early-late",
  use_dominators = true,
  insert_anti_deps = true
>
```

**Why This Band Third:**
Scheduling is where abstract IR meets concrete execution order. This is THE convergence point Click emphasized.

**Critical Insight:**
Band 3 is the first point where Sea of Nodes differs significantly from traditional SSA:
- **Traditional SSA:** Instruction order implicit in basic block
- **Sea of Nodes:** Instruction order computed via scheduling

This enables more aggressive optimization because nodes can float freely until scheduling pins them down.

---

### Band 4: Types → Type Systems + Lattice Analysis

**Sea of Nodes Concept:**
- Type lattice (TOP → Constants → BOTTOM)
- Nullable vs non-nullable references
- Narrow types with range analysis

**Real-World IR Equivalent:**
```swift
// Swift's type system (similar philosophy)
var x: Int?  // Nullable
var y: Int   // Non-nullable

// LLVM + Type Metadata
%struct.Foo = type { i32, float }
%1 = load %struct.Foo, ptr %p, !nonnull !0
```

**Dovetail Points:**
- Type lattice = Abstract interpretation (Cousot & Cousot)
- Nullable types = Rust/Swift null safety
- Range analysis = LLVM's ConstantRange
- Type meet = Type unification (Hindley-Milner)

**MLIR Alignment:**
```mlir
// MLIR type system
!range_int = !int<range=[0,100]>      // Range-constrained int
!nullable_ptr = !ptr<nullable>         // Nullable pointer
!array = !tensor<10xi32>               // Fixed array
```

**Why This Band Fourth:**
Can't optimize what you don't understand. Types enable:
- Null check elimination (nullable analysis)
- Bounds check elimination (range analysis)
- Devirtualization (reference analysis)
- Vectorization (type-width analysis)

---

### Band 5+: Advanced Opts → Production Compiler Passes

**Sea of Nodes Concept:**
- Escape analysis
- Inlining
- Loop optimizations
- Vectorization

**Real-World IR Equivalent:**
```
; LLVM Optimization Pass Pipeline
- EarlyCSE (common subexpression elimination)
- SROA (scalar replacement of aggregates)
- InlineFunction (function inlining)
- LICM (loop invariant code motion)
- LoopVectorize (auto-vectorization)
- MemCpyOpt (memory intrinsic optimization)
```

**Dovetail Points:**
Each Band 5+ chapter maps to a proven LLVM pass:
- **Escape analysis** → LLVM's capture tracking
- **Inlining** → LLVM's function inliner
- **Loop opts** → LLVM's LoopPass infrastructure
- **Vectorization** → LLVM's SLP/Loop vectorizers

**MLIR Alignment:**
Each optimization becomes an MLIR dialect transformation:
```mlir
// MLIR transformation patterns
transform.sequence {
  %0 = transform.structured.vectorize %arg0  // Vectorization
  %1 = transform.loop.fusion %0              // Loop fusion
  %2 = transform.inline %1                   // Inlining
}
```

---

## The Strategic Choice: Why These Bands?

### Alignment with Production Compilers

1. **Band 1** = SSA (universal)
2. **Band 2** = CFG + Memory Model (universal)
3. **Band 3** = Scheduling (Click's innovation, now LLVM MachineScheduler)
4. **Band 4** = Rich type system (Rust/Swift/MLIR philosophy)
5. **Band 5+** = Standard opt passes (LLVM pipeline)

### Enabling Subsumption Engine

The bands were chosen so subsumption can query real compiler concepts:

```cpp
// Each band adds a queryable dimension
auto optimizable = engine.query()
    .whereCFG(dominatedBy(loop))           // Band 2/3 concept
    .whereType(subsumes<NumericType>())    // Band 4 concept
    .whereSchedule(hoistable())            // Band 3 concept
    .whereEscape(doesNotEscape())          // Band 5 concept
    .projectToMLIR(VectorDialect);         // MLIR lowering
```

### Dovetailing = Interoperability

Because bands align with real IR concepts, we can:

**Import from LLVM:**
```cpp
// LLVM IR → Sea of Nodes
auto* soaGraph = importFromLLVM(llvmModule);
soaGraph.optimize();  // Use Sea of Nodes opts
auto* newLLVM = exportToLLVM(soaGraph);
```

**Export to MLIR:**
```cpp
// Sea of Nodes → MLIR
auto* mlirModule = soaGraph.lowerToMLIR();
mlirModule.applyPatterns();  // Use MLIR transforms
auto* llvmIR = mlirModule.lowerToLLVM();
```

**Bidirectional Flow:**
```
Source Code
    ↓
Sea of Nodes (Band 1-4)
    ↓
MLIR Dialects (pattern matching)
    ↓
LLVM IR (backend)
    ↓
Machine Code
```

---

## Real-World Validation

### Production Compilers Using Similar Concepts

| Compiler | IR Concept | Band Equivalent |
|----------|-----------|-----------------|
| **HotSpot C2** | Sea of Nodes | Bands 1-4 (Click's original) |
| **LLVM** | SSA + DominatorTree | Bands 1-3 |
| **GCC** | GIMPLE + dominators | Bands 1-3 |
| **V8 TurboFan** | Sea of Nodes variant | Bands 1-4 |
| **MLIR** | Multi-level dialects | All bands (target) |
| **Rust rustc** | MIR (memory-SSA) | Bands 1-2 |

### Academic Foundations

Each band concept has academic pedigree:

- **Band 1 SSA:** Cytron et al. 1991
- **Band 2 Memory SSA:** Chow et al. 1996
- **Band 3 Scheduling:** Click & Cooper 1995
- **Band 4 Lattice Types:** Cousot & Cousot 1977 (abstract interpretation)
- **Band 5+ Opts:** Decades of PL research

---

## Why This Matters for cppfort

### 1. MLIR Integration Path

Bands map cleanly to MLIR dialects:
- Band 1-2 → `arith`, `cf` dialects
- Band 3 → Scheduling attributes
- Band 4 → `memref`, `tensor` types
- Band 5+ → `vector`, `affine`, `linalg` dialects

### 2. CPP2 Semantics Preservation

CPP2 has specific IR concepts:
- Definite initialization (Band 1-2)
- Lifetime analysis (Band 2)
- Move semantics (Band 2 + escape analysis)
- Bounds safety (Band 4)

Bands align perfectly.

### 3. Multi-Language Target

Because bands align with universal concepts, stage0 can emit:
- **C:** Via LLVM IR lowering
- **C++:** Direct emission (Band 4 types)
- **Rust:** Via MLIR lowering
- **WASM:** Via MLIR WASM dialect
- **GPU:** Via MLIR GPU dialect

### 4. Validation Strategy

Each band has established correctness criteria:
- **Band 1:** SSA dominance property
- **Band 2:** Memory consistency model
- **Band 3:** Scheduling legality (dominators respected)
- **Band 4:** Type soundness
- **Band 5+:** Semantics preservation

These aren't invented - they're textbook compiler correctness.

---

## Implementation Notes

### Why Not Just Use LLVM IR Directly?

**Sea of Nodes advantages:**
1. **More optimization freedom** - Nodes float until scheduled
2. **Simpler peephole rules** - Local transformations, no CFG rewrites
3. **Type lattice integration** - Types and values unified
4. **Natural for JIT** - Graph construction is incremental
5. **MLIR integration** - Better match for dialect lowering

**But we dovetail to reuse:**
- LLVM's backend (register allocation, instruction selection)
- MLIR's dialects (vector, GPU, etc.)
- Existing optimization algorithms (proven correct)

### The Band Structure Enables Hybrid Approach

```
Source → Sea of Nodes (Bands 1-4) → MLIR (pattern matching) → LLVM (backend)
         [High-level opts]          [Dialect lowering]         [Code gen]
```

Each phase uses the IR concept that's strongest for that task.

---

## Conclusion

**The bands aren't just pedagogical organization - they're strategic alignment with proven compiler IR concepts.**

This enables:
- ✅ Reuse of established algorithms
- ✅ Interoperability with LLVM/MLIR
- ✅ Validation against known correctness criteria
- ✅ Incremental implementation with testable milestones
- ✅ Subsumption queries over real compiler abstractions

**The subsumption engine queries concepts that have 30+ years of research and production validation.**

This is why the bands were chosen specifically to dovetail with real-world frontend IR - not academic purity, but pragmatic engineering.
