# JIT Architecture: MLIR Pipeline and IR Lowering

## Overview

cppfort uses a **multi-stage IR pipeline** with MLIR as the optimization infrastructure. The key design principle is:

> **Semantic information flows FROM the AST THROUGH MLIR to the codegen - we're NOT lowering SON, we're passing SON-optimized code through to C++ generation.**

## Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CPP2 SOURCE CODE                              │
│                              main.cpp2                                     │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                PARSER (selfhost rbcursive.cpp2 via cppfront)                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Feature-Stream Scanning → Pure cpp2 bootstrap parsing → Smoke gate   │  │
│  │   (rbcursive.cpp2)        (selfhost surface)            (ctest)      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC ANALYSIS (semantic_analyzer.cpp)               │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Type Checking → Escape Analysis → Borrow Checking → Lifetime       │  │
│  │                  (Phase 1-2)            (Phase 2)     (Phase 2)     │  │
│  │                                                                     │  │
│  │  ATTACHES SemanticInfo to AST nodes:                               │  │
│  │  - escape_info: EscapeKind (NoEscape, EscapeToReturn, etc.)         │  │
│  │  - borrow: OwnershipKind (Owned, Borrowed, MutBorrowed, Moved)     │  │
│  │  - memory_transfer: GPU/DMA transfer tracking                       │  │
│  │  - channel_transfer: Send/recv ownership tracking                  │  │
│  │  - arena: ArenaRegion (JIT memory pool)                            │  │
│  │  - coroutine_frame: CoroutineFrameStrategy                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AST → FIR LOWERING                                  │
│                    (ast_to_fir.cpp - future work)                           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Convert AST nodes to Cpp2FIR dialect operations                     │  │
│  │                                                                     │  │
│  │  AST::VariableDeclaration → cpp2fir.var {                          │  │
│  │    escape = #cpp2fir.escape<"no_escape">,                          │  │
│  │    arena_scope = #cpp2fir.arena_scope<1>                           │  │
│  │  }                                                                  │  │
│  │                                                                     │  │
│  │  AST::FunctionDeclaration → cpp2fir.func                            │  │
│  │  AST::BinaryExpression → cpp2fir.add, cpp2fir.mul, etc.            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FIR: Cpp2 Front-IR (Cpp2FIRDialect.td)                │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ HIGH-LEVEL SEMANTIC OPERATIONS                                       │  │
│  │                                                                     │  │
│  │  Operations: var, func, add, mul, call, if, while, etc.          │  │
│  │  Attributes:                                                        │  │
│  │    - #cpp2fir.escape<"no_escape|heap|return|param|global|...">      │  │
│  │    - #cpp2fir.arena_scope<scope_id>                                 │  │
│  │    - !cpp2.arena<scope_id, pointee_type>                            │  │
│  │                                                                     │  │
│  │  FIR is ANNOTATED with semantic info from AST analysis              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
┌──────────────────────────┐    ┌─────────────────────────────────────────────┐
│  OPTIMIZATION PASSES     │    │  JIT ALLOCATION PASSES (Phase 7-10)        │
│  (FIR-level analysis)    │    │  Run ON FIR, NOT lowering to SON          │
├──────────────────────────┤    ├─────────────────────────────────────────────┤
│ • FIRTransferElimination │    │ • FIRArenaInferencePass                    │
│   (Phase 3)              │    │   - Analyzes NoEscape aggregates           │
│   - Uses escape attrs    │    │   - Assigns arena scope IDs                │
│   - Removes GPU/DMA      │    │   - Tags with #cpp2fir.arena_scope        │
│     transfers for NoEsc  │    │                                             │
│                          │    │ • FIRCoroutineFrameSROAPass                 │
│ • FIRDMASafety (Phase 3) │    │   - Detects non-escaping coroutines         │
│   - Validates DMA        │    │   - Tags with coroutine_frame attr         │
│     safety               │    │                                             │
│                          │    │ KEY: These runs on FIR DIRECTLY            │
│                          │    │ NO lowering to SON required!               │
└──────────────────────────┘    └─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 FIR → SON LOWERING (ConvertFIRToSON.cpp)                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Pattern-based dialect conversion:                                    │  │
│  │                                                                     │  │
│  │  cpp2fir.add → sond.add                                            │  │
│  │  cpp2fir.mul → sond.mul                                            │  │
│  │  cpp2fir.func → sond.func                                          │  │
│  │  ... (structural conversion, preserving semantics)                   │  │
│  │                                                                     │  │
│  │  SEMANTIC ATTRIBUTES are PRESERVED:                                 │  │
│  │  - Escape analysis info → passed through                           │  │
│  │  - Arena annotations → attached to SON ops                         │  │
│  │  - Memory transfer info → preserved                                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SON: Sea of Nodes (Cpp2SONDialect.td)                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ LOW-LEVEL OPTIMIZABLE IR                                             │  │
│  │                                                                     │  │
│  │  Based on Chapter 24 "Sea of Nodes" design:                        │  │
│  │  - CFG Nodes: Start, Stop, Region, Loop, If, CProj                │  │
│  │  - Data Nodes: Add, Mul, Div, Phi, etc.                           │  │
│  │  - Types: Control, Memory, Integer (with lattices)                │  │
│  │                                                                     │  │
│  │  UNIFIED control+data flow representation                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   SON OPTIMIZATION PASSES                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  • SCCP (Sparse Conditional Constant Propagation)                       │  │
│    - Optimistic type propagation from TOP→BOTTOM                       │  │
│    - Interprocedural analysis                                          │  │
│    - Uses type lattice for monotone framework                          │  │
│                                                                     │  │
│  • IterPeeps (Iterative Peephole)                                      │  │
│    - Random worklist order for coverage                                │  │
│    - Applies fold() and idealize() until fixed point                   │  │
│                                                                     │  │
│  • DCE (Dead Code Elimination)                                         │  │
│  • CSE (Common Subexpression Elimination)                              │  │
│  • Loop optimizations                                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              OPTIMIZED SON (still has semantic annotations!)                │
│                                                                             │
│  IMPORTANT: SON is NOT lowered further! We go DIRECTLY to codegen.         │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  CODE GENERATION (code_generator.cpp)                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ Reads SEMANTIC INFO (from AST or preserved in IR)                   │  │
│  │ to make JIT allocation decisions:                                   │  │
│  │                                                                     │  │
│  │  determine_allocation_strategy(VariableDeclaration):               │  │
│  │    if (semantic_info.arena) → Arena                                │  │
│  │    if (escape.kind == NoEscape && aggregate) → Arena               │  │
│  │    if (escape.kind == Escaping) → Heap                              │  │
│  │    else → Stack                                                    │  │
│  │                                                                     │  │
│  │  Generates C++ with allocation comments:                           │  │
│  │    int x(42);  // Allocation: stack (NoEscape local)                │  │
│  │    cpp2::arena_alloc<std::vector>(arena<1>(), {})                  │  │
│  │      // Allocation: arena scope 1 (NoEscape aggregate)             │  │
│  │    std::make_unique<T>({})  // Allocation: heap (escaping)         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           C++ OUTPUT CODE                                  │
│                          optimized.cpp                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. **We Pass Through SON, We Don't Lower It**

The SON (Sea of Nodes) dialect is the **final optimization stage**. After SON optimizations (SCCP, CSE, peephole), we **do NOT** lower to LLVM IR or another backend. Instead:

- Optimized SON operations → C++ code generation
- Semantic annotations preserved → Inform allocation decisions
- Type lattice information (from SCCP) → Used for constant folding

### 2. **JIT Passes Run on FIR, Not SON**

The JIT allocation passes (Phase 7-10) run **directly on FIR** before SON lowering:

```md
FIR (with semantic annotations)
  → FIRArenaInferencePass
  → FIRCoroutineFrameSROAPass
  → Convert to SON (preserves annotations)
  → SON optimizations
  → Codegen (uses annotations for allocation)
```

This design keeps semantic information **close to the source** where it's most accurate.

### 3. **Semantic Info is Source of Truth**

The `SemanticInfo` struct in `ast.hpp` is the **authoritative source** for:

- Escape analysis results
- Borrow/ownership state
- Memory transfer requirements
- Arena allocation decisions
- Coroutine frame strategies

MLIR IRs (FIR and SON) **carry** this information but don't recompute it.

### 4. **Allocation Strategy is Final Decision**

The `determine_allocation_strategy()` function in `code_generator.cpp` makes the **final call**:

```cpp
AllocationStrategy determine_allocation_strategy(VariableDeclaration* decl) {
    // Priority order:
    // 1. Explicit arena annotation (from analysis pass)
    if (decl->semantic_info.arena) return Arena;

    // 2. Coroutine frame strategy
    if (decl->semantic_info.coroutine_frame)
        return convert(coroutine_frame.strategy);

    // 3. Escape analysis
    if (decl->escape_info) {
        switch (escape.kind) {
            case NoEscape:
                return type_is_aggregate ? Arena : Stack;
            case Escaping:
                return Heap;
        }
    }

    // 4. Default: stack
    return Stack;
}
```

## Data Flow Example

```cpp
// Cpp2 source
main: () = {
    data: std::vector<int> = std::vector<int>{1, 2, 3};
    print(data);
}

// AST (after semantic analysis)
VariableDeclaration {
    name: "data"
    escape_info: EscapeInfo { kind: NoEscape }
    semantic_info: SemanticInfo {
        escape: EscapeInfo { kind: NoEscape }
        arena: ArenaRegion { scope_id: 1 }
    }
}

// FIR (with annotations)
%data = cpp2fir.var [...] {
    escape = #cpp2fir.escape<"no_escape">,
    arena_scope = #cpp2fir.arena_scope<1>
}

// After ArenaInferencePass
%data = cpp2fir.var [...] {
    escape = #cpp2fir.escape<"no_escape">,
    arena_scope = #cpp2fir.arena_scope<1>  // Confirmed
}

// SON (after conversion, annotations preserved)
%data = sond.var [...] {
    escape = #cpp2fir.escape<"no_escape">,
    arena_scope = #cpp2fir.arena_scope<1>
}

// Codegen (reads semantic_info)
using Arena_1 = cpp2::monotonic_arena<1>;
Arena_1 arena_1;
auto data = cpp2::arena_alloc<std::vector<int>>(arena_1(), ...)
  // Allocation: arena scope 1 (NoEscape aggregate)
```

## Why This Architecture?

1. **Semantic Preservation**: Escape/ownership analysis happens once on AST, not recomputed in MLIR
2. **Separation of Concerns**: MLIR handles optimization, AST handles semantics, codegen handles allocation
3. **Debuggability**: Each stage has clear responsibility and can be inspected independently
4. **Incremental Adoption**: Can enable/disable stages without breaking pipeline
5. **No Re-Lowering**: SON is the final IR - no need for LLVM IR intermediate

## Future Work

- [ ] Full AST → FIR lowering implementation
- [ ] SON → C++ codegen (currently uses AST directly)
- [ ] MLIR JIT execution (compile FIR/SON to machine code at runtime)
- [ ] Declarative allocation rules (move strategy logic from C++ to MLIR patterns)
