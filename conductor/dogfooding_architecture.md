# cppfort Dogfooding Architecture

## The Closed Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                   │
│  src/selfhost/rbcursive.cpp2          (parser combinators)      │
│  src/selfhost/trikeshed_join.cpp2     (Join<A,B> type)          │
│  src/selfhost/trikeshed_series.cpp2   (Series<T> type)          │
│  src/selfhost/trikeshed_manifold.cpp2 (Coordinates, Manifold)   │
│  src/selfhost/trikeshed_either.cpp2   (Either<L,R> type)        │
│  src/selfhost/canonical_types.cpp2    (SoN type definitions)    │
│  src/selfhost/bootstrap_tags.cpp2     (canonical node tags)     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 1: BOOTSTRAP                           │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   cpp2      │───▶│   cppfront  │───▶│   C++20     │         │
│  │   source    │    │  (bridge)   │    │   output    │         │
│  └─────────────┘    └─────────────┘    └──────┬──────┘         │
│                                               │                 │
│                                               ▼                 │
│                                        ┌─────────────┐         │
│                                        │  clang++    │         │
│                                        │  (compile)  │         │
│                                        └──────┬──────┘         │
│                                               │                 │
│                                               ▼                 │
│                                        ┌─────────────┐         │
│                                        │  cppfort    │         │
│                                        │  (bootstrap)│         │
│                                        └─────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ (proves cpp2 → executable works)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 2: SELF-HOSTED                           │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   cpp2      │───▶│   cppfort   │───▶│   MLIR SoN  │         │
│  │   source    │    │ (self-host) │    │   dialect   │         │
│  └─────────────┘    └──────┬──────┘    └──────┬──────┘         │
│                            │                    │               │
│                            │   canonical AST    │               │
│                            │   normalization    │               │
│                            │                    │               │
│                            ▼                    ▼               │
│                     ┌─────────────────────────────────┐        │
│                     │      SoN Optimization          │        │
│                     │  - Constant propagation        │        │
│                     │  - Zero-cost abstraction       │        │
│                     │  - Alias analysis              │        │
│                     └──────────────┬────────────────┘        │
│                                    │                          │
│                                    ▼                          │
│                     ┌─────────────────────────────────┐        │
│                     │   MLIR Lowering (LLVM IR)      │        │
│                     │   - memref → LLVM arrays       │        │
│                     │   - SoN ops → LLVM intrinsics   │        │
│                     └──────────────┬────────────────┘        │
│                                    │                          │
│                                    ▼                          │
│                     ┌─────────────────────────────────┐        │
│                     │   Executable (cppfort2)        │        │
│                     └─────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ (the dogfood test)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     THE CLOSED LOOP                             │
│                                                                 │
│   cppfort compiles src/selfhost/*.cpp2 → cppfort2              │
│                                                                 │
│   cppfort2 compiles src/selfhost/*.cpp2 → cppfort3             │
│                                                                 │
│   cppfort2 == cppfort3  (bitwise identical output)             │
│                                                                 │
│   ✓ Self-hosting achieved                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Critical Path

### 1. Parser Surface (rbcursive.cpp2)
**Status:** 3686 lines, parses cpp2-like syntax
**Gap:** AST construction, error recovery

```cpp2
// Current: parses, emits features
scan_result: @struct <T: type> type = {
    outcome: scan_signal = ();
    value: std::optional<T> = ();
    consumed: int = 0;
}

// Need: builds canonical AST nodes
```

### 2. Canonical Types (trikeshed_*.cpp2)
**Status:** Core types defined
**Gap:** SoN dialect integration

```cpp2
// Current: cpp2 structs
join: @struct <A: type, B: type> type = {
    a: A = ();
    b: B = ();
}

// Need: Maps to SoN operations
// cpp2.join<A, B> → SoN dialect
```

### 3. SoN Dialect (include/cpp2/)
**Status:** TableGen defined
**Gap:** Build integration, pass pipeline

```tablegen
// cpp2_ops.td
def JoinOp : Cpp2_Op<"join"> {
    let arguments = (ins AnyType:$a, AnyType:$b);
    let results = (outs JoinType:$result);
}
```

### 4. Bootstrap Bridge
**Current:** cppfront transpiles cpp2 → C++
**Target:** cppfort parses cpp2 → SoN → LLVM

## The Missing Links

| Component | Current | Dogfood Target |
|-----------|---------|----------------|
| Parser | Feature stream | Canonical AST builder |
| Types | cpp2 structs | SoN type constraints |
| Dialect | TableGen only | Built, linked, passes active |
| Lowering | cppfront | cppfort native |
| Verification | Manual | Self-compilation test |

## First Acceptance Gate

**Minimal self-host:**
```bash
# Stage 1: cppfront bridge
cppfront src/selfhost/bootstrap_tags.cpp2 -o bootstrap_tags.cpp
clang++ bootstrap_tags.cpp -o bootstrap_tags
./bootstrap_tags  # runs, prints tags

# Stage 2: cppfort self-host
cppfort src/selfhost/bootstrap_tags.cpp2 -o bootstrap_tags2
./bootstrap_tags2  # identical output
```

**Next gate:** rbcursive compiles itself
```bash
cppfort src/selfhost/rbcursive.cpp2 -o rbcursive2
diff <(./rbcursive2 test.cpp2) <(cppfront test.cpp2)
# identical output
```

**Final gate:** Full self-host
```bash
cppfort src/selfhost/*.cpp2 -o cppfort2
cppfort2 src/selfhost/*.cpp2 -o cppfort3
diff cppfort2 cppfort3
# identical binaries
```

## Architecture Principles

1. **Semantic objects first**: `join<A,B>`, `series<T>` are canonical
2. **Zero-cost abstraction**: SoN optimization proves no overhead
3. **Surface normalization**: `j(a,b)` → `join<A,B>` node → optimal code
4. **Closed loop**: cppfort eats its own cpp2 source
5. **TrikeShed lineage**: Kotlin semantics, cpp2 syntax, SoN optimization

## Current Blockers

1. SoN dialect not in build
2. No AST construction from rbcursive
3. No lowering pipeline (cpp2 → SoN → LLVM)
4. No self-compilation test

## Next Actions

1. Enable SoN dialect in CMake
2. Add AST node builders to rbcursive
3. Wire parser → SoN → LLVM
4. Bootstrap test: compile simplest cpp2 file
5. Iterate to full self-host

---

*This is the dogfood deliverable: cppfort compiling itself via canonical SoN representation, proving the bootstrap loop closes.*