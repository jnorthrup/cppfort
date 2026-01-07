# Semantic AST Enhancements for Cpp2

**Date**: 2025-12-28
**Objective**: Strengthen semantic intent preservation in Cpp2 AST with support for escape analysis, borrowing, external RAM, and channelized concurrency

## Current State

### Cpp2 AST Capabilities (as implemented)
- **Parameter qualifiers**: `in`, `out`, `inout`, `move`, `forward`
- **Concurrency primitives**: Channels, coroutine scopes, spawn/await
- **Memory regions**: GPU global/shared/constant, DMA buffers, local compute
- **Kernel annotations**: GPU kernel functions with launch config
- **Parallel loops**: GPU kernel conversion via `parallel_for`

### Gaps Identified from pure2-hello.cpp2 Test

Cppfort transpilation of `pure2-hello.cpp2` revealed semantic losses:

1. **Parameter passing semantics lost**:
   - Source Cpp2: `decorate: (inout s: std::string) = { ... }`
   - Cppfort output: `void decorate(std::string s)` ❌ (by-value)
   - Cppfront reference: `auto decorate(std::string& s) -> void` ✓ (by-reference)
   - **Impact**: `inout` semantics erased - mutations don't propagate

2. **No escape analysis tracking**:
   - Variables escaping local scope not annotated
   - Lifetime information lost in transpilation
   - Cannot verify safety contracts at compile time

3. **Borrowing semantics not represented**:
   - No distinction between owned values and borrowed references
   - Rust-like borrow checking impossible
   - Aliasing information unavailable

4. **External memory pipeline semantics incomplete**:
   - Memory region annotations exist but not connected to escape analysis
   - DMA transfers not tracked for SSA state escape
   - No super-optimization for lifecycle-based optimizations

5. **Channel semantics not integrated**:
   - Channels defined but not tied to ownership analysis
   - Send/receive not tracked for escape
   - No integration with concurrency safety checks

## Proposed Enhancements

### 1. Escape Analysis Framework

**Goal**: Track where values escape local scope and their lifetimes

**AST Additions**:
```cpp
enum class EscapeKind {
    NoEscape,          // Value stays local (stack)
    EscapeToHeap,      // Stored in heap-allocated object
    EscapeToReturn,    // Returned from function
    EscapeToParam,     // Stored via pointer/reference parameter
    EscapeToGlobal,    // Stored in global variable
    EscapeToChannel,   // Sent through channel
    EscapeToGPU,       // Transferred to GPU memory
    EscapeToDMA        // Transferred via DMA buffer
};

struct EscapeInfo {
    EscapeKind kind;
    std::vector<ASTNode*> escape_points;  // Where value escapes
    std::optional<MemoryRegion> dest_region;  // Destination memory region
    bool needs_lifetime_extension;
};

class EscapeAnalysis {
    std::unordered_map<VarDecl*, EscapeInfo> escape_map;

    EscapeInfo analyze_variable(VarDecl* var);
    void propagate_escape_through_calls();
    void verify_borrow_safety();
};
```

**Integration Points**:
- Attach `EscapeInfo` to every `VarDecl` in AST
- Compute during semantic analysis after type checking
- Use for optimizing memory allocation (stack vs heap)
- Validate safety contracts (no-escape guarantees)

### 2. Borrowing and Ownership Tracking

**Goal**: Rust-like ownership semantics with borrow checking

**AST Additions**:
```cpp
enum class OwnershipKind {
    Owned,      // Unique owner (value semantics)
    Borrowed,   // Immutable borrow (shared reference)
    MutBorrowed,// Mutable borrow (exclusive reference)
    Moved       // Ownership transferred
};

struct BorrowInfo {
    OwnershipKind kind;
    ASTNode* owner;  // Original owner if borrowed
    std::vector<ASTNode*> active_borrows;  // Live borrows
    LifetimeRegion lifetime;
};

struct LifetimeRegion {
    ASTNode* scope_start;
    ASTNode* scope_end;
    std::vector<LifetimeRegion*> nested_regions;
    bool outlives(const LifetimeRegion& other) const;
};

class BorrowChecker {
    void check_no_aliasing_violations();
    void check_borrow_outlives_owner();
    void check_move_invalidates_borrows();
    void enforce_exclusive_mut_borrow();
};
```

**Parameter Qualifier Mapping**:
```
in       → Borrowed      (immutable borrow)
out      → MutBorrowed   (mutable borrow, no read before write)
inout    → MutBorrowed   (mutable borrow with read/write)
move     → Moved         (ownership transfer)
forward  → Moved/Borrowed (conditional based on argument)
```

**Enforcement**:
- `in`: Cannot mutate parameter, can have multiple simultaneous borrows
- `out`: Must write before function returns, exclusive access
- `inout`: Can read and write, exclusive access during function call
- `move`: Original variable invalidated after call
- `forward`: Ownership transferred if rvalue, borrowed if lvalue

### 3. External Memory Pipeline Integration

**Goal**: Lifecycle-based super-optimization for SSA state escape avoidance

**Current FIR Dialect Memory Ops**:
```mlir
// From Cpp2FIRDialect.td
cpp2fir.mem_region "gpu_global" size 1024 dma -> !cpp2.handle
cpp2fir.kernel @kernel_name() launch "grid(256,256) block(32)"
               memory "streaming" { ... }
```

**Enhanced Escape Integration**:
```cpp
struct MemoryTransfer {
    EscapeKind escape_kind;
    MemoryRegion* source_region;
    MemoryRegion* dest_region;
    bool is_async;  // DMA transfer
    std::vector<VarDecl*> transferred_vars;
    LifetimeRegion transfer_lifetime;
};

class ExternalMemoryAnalysis {
    // Track when SSA values escape to external memory
    std::vector<MemoryTransfer> track_gpu_transfers();

    // Optimize away transfers for local-only values
    void eliminate_unnecessary_transfers();

    // Super-optimize based on lifetime analysis
    void apply_lifecycle_optimizations();

    // Validate DMA safety
    void verify_dma_no_aliasing();
};
```

**Optimization Example**:
```cpp
// Cpp2 code
kernel_func: () = {
    x: int = 42;           // EscapeAnalysis: NoEscape
    y: int = gpu_compute(x); // EscapeAnalysis: EscapeToGPU

    // Optimization: x stays on stack, no transfer needed
    // Only y transferred to/from GPU
}
```

### 4. Channelized Concurrency Integration

**Goal**: Integrate channel operations with escape and ownership analysis

**Current FIR Dialect Channel Ops**:
```mlir
cpp2fir.channel !cpp2.struct<"int"> buffer 10 -> !cpp2.channel<int>
cpp2fir.send %channel, %value : !cpp2.channel<int>, !cpp2.struct<"int">
cpp2fir.recv %channel : !cpp2.channel<int> -> !cpp2.struct<"int">, i1
```

**Enhanced Semantic Tracking**:
```cpp
struct ChannelTransfer {
    EscapeKind escape_kind = EscapeKind::EscapeToChannel;
    ASTNode* send_point;
    ASTNode* recv_point;  // May be unknown
    OwnershipKind ownership_transfer;  // Move or Borrow
    LifetimeRegion channel_lifetime;
};

class ConcurrencyAnalysis {
    // Track values sent through channels
    std::unordered_map<VarDecl*, ChannelTransfer> channel_escapes;

    // Verify channel safety
    void check_no_data_race();
    void check_channel_lifetime_bounds();
    void check_send_ownership_transfer();

    // Optimize channel usage
    void eliminate_redundant_sends();
    void batch_channel_transfers();
};
```

**Ownership Rules for Channels**:
```cpp
// Send with move semantics (default)
ch.send(move value);  // value invalidated after send
// → OwnershipKind::Moved

// Send with borrow (copy)
ch.send(in value);    // value copied, original remains valid
// → OwnershipKind::Borrowed

// Receive (always move)
result := ch.recv();  // result is new owner
// → OwnershipKind::Owned
```

### 5. Unified Semantic Representation

**Goal**: Single AST representation capturing all semantic information

```cpp
struct SemanticInfo {
    // Ownership and borrowing
    BorrowInfo borrow;

    // Escape analysis
    EscapeInfo escape;

    // Memory location
    std::optional<MemoryRegion> memory_region;
    std::optional<MemoryTransfer> active_transfer;

    // Concurrency
    std::optional<ChannelTransfer> channel_transfer;
    std::optional<KernelLaunch> kernel_context;

    // Lifetime bounds
    LifetimeRegion lifetime;
    std::vector<LifetimeRegion*> must_outlive;

    // Safety contracts
    std::vector<SafetyContract> contracts;
};

// Attach to every ASTNode
class ASTNode {
    std::unique_ptr<SemanticInfo> semantic_info;

    bool is_safe() const;
    bool can_optimize_away() const;
    std::string explain_semantics() const;
};
```

## Expected Outcomes

### Semantic Preservation
- **Parameter semantics**: 100% accurate (in/out/inout/move/forward)
- **Escape analysis**: All escape points tracked
- **Borrowing**: Rust-like safety guarantees
- **Memory regions**: Correct DMA and GPU transfer tracking
- **Concurrency**: Channel ownership and data race detection

### Optimization Opportunities
- **Stack allocation**: NoEscape values stay on stack
- **Transfer elimination**: Avoid unnecessary GPU/DMA transfers
- **Channel batching**: Combine multiple sends
- **Lifetime extension**: Minimize heap allocations

### Safety Guarantees
- **No aliasing violations**: Exclusive mutable borrows enforced
- **No use-after-move**: Moved variables invalidated
- **No data races**: Channel operations validated
- **DMA safety**: No aliasing during async transfers

## Integration with Existing Infrastructure

### Corpus Regression Tests
- Enhanced tests will exercise new semantic analysis
- Compare cppfort output against cppfront reference
- Measure semantic loss reduction (target: >50% improvement)

### MLIR Lowering
- Preserve `SemanticInfo` during AST→MLIR conversion
- Emit `!cpp2.owned`, `!cpp2.borrowed` type attributes
- Add lifetime annotations to MLIR operations
- Enable MLIR-level optimization based on semantics

## References

- **Rust ownership**: https://doc.rust-lang.org/book/ch04-00-understanding-ownership.html
- **Escape analysis**: https://en.wikipedia.org/wiki/Escape_analysis
- **DMA safety**: CUDA programming guide, section on async memory copies
- **CSP channels**: https://en.wikipedia.org/wiki/Communicating_sequential_processes
- **Cpp2 contracts**: https://github.com/hsutter/cppfront (safety philosophy)

## Metrics

| Metric | Before | Target | Status |
|--------|--------|--------|--------|
| Parameter semantic accuracy | 0% (lost inout) | 100% | ⏳ Pending |
| Escape analysis coverage | 0% | 100% | ⏳ Pending |
| Borrowing enforcement | 0% | 100% | ⏳ Pending |
| Memory transfer tracking | 0% | 100% | ⏳ Pending |
| Channel safety validation | 0% | 100% | ⏳ Pending |
| Average corpus semantic loss | 1.0 (max) | <0.15 | ⏳ Pending |
| pure2-hello.cpp2 loss | 1.0 | <0.05 | ⏳ Pending |

---

## Extended Scope: JIT Memory-Managed Front-IR Architecture (Phases 7-10)

**Date Extended**: 2026-01-06
**Objective**: Leverage cppfront's ownership semantics + C++26 reflection to build a lifetime-aware Front-IR that offloads allocation decisions to the JIT's analysis pass

### 6. Scope-Inferred Arena Allocation (Phase 7)

**Goal**: Automatic arena allocation for non-escaping scopes based on escape analysis

**Architecture**:
```cpp
// Cpp2 source with implicit arena binding
func: (data: vector<widget>) -> result = {
    // Front-IR tags this scope → JIT allocates monotonic arena
    // All allocations demoted from heap to arena-bump
}
```

**Implementation**:
- `ArenaRegion` linked to `LifetimeRegion` (scope-based memory regions)
- JIT pass: `ArenaInferencePass` analyzes escape-annotated FIR
- MLIR attributes: `#cpp2.arena_scope<scopeID>`, `!cpp2.arena<scopeID>`
- Decision logic: NoEscape aggregates → arena-allocated (not heap)

**Expected Outcome**: Local vectors/maps/strings use monotonic arena allocation; heap reserved for escaping values

---

### 7. Coroutine Frame Elision (Phase 8)

**Goal**: SROA (Scalar Replacement of Aggregates) on coroutine frames when escape analysis proves non-escaping

**Kotlin-style Structured Concurrency Port**:
- `CoroutineScope` semantics → C++26 `std::execution` senders/receivers
- `launch {}` blocks → `std::execution::schedule`
- `async/await` → sender/receiver chains

**MLIR Pass**: `CoroutineFrameSROA`
- Detects non-escaping coroutine frames (suspended state contained within parent scope)
- Converts heap-boxed frames → stack-pinned or arena-slotted
- Attribute: `#cpp2.coroutine_frame<stack|arena|heap>`

**Expected Outcome**: Structured concurrency coroutines avoid heap allocation; deterministic frame teardown

---

### 8. C++26 Integration (Phase 9)

**Goal**: Synergize C++26 features with lifetime-aware Front-IR

**C++26 Features Leveraged**:

1. **Reflection (`std::meta`)**:
   - Compute `std::inplace_vector<T, N>` capacity at compile time
   - `reflection_driven_sbo_size()` utility for optimal SBO sizing
   - Attribute: `#cpp2.reflection_sized<bytes>`

2. **Contracts (`[[expects]]`, `[[ensures]]`)**:
   - Feed preconditions into alias analysis
   - Derive no-alias guarantees from contract annotations
   - Add `ContractInfo` to `SemanticInfo`
   - Attribute: `#cpp2.contract<precondition|postcondition>`

3. **Pattern Matching (`inspect`)**:
   - Exhaustive resource state tracking
   - `ResourceState` enum: Uninitialized, Initialized, Moved, Borrowed
   - Track state transitions through `inspect` branches

**Expected Outcome**: Contracts strengthen alias analysis; reflection optimizes inplace storage; pattern matching ensures exhaustive state coverage

---

### 9. JIT Codegen Backend (Phase 10)

**Goal**: Final JIT pass emits C++ allocators based on lifetime-aware Front-IR annotations

**Analytical Pass Pipeline**:
```
Source → cppfront lowering → Lifetime-annotated IR →
  Escape analysis → Region inference → JIT codegen (stack/arena/rare-heap)
```

**AllocationStrategyPass**:
- Input: Annotated FIR with `#cpp2.arena_scope`, `!cpp2.arena<scopeID>`, `#cpp2.coroutine_frame`
- Output: C++ code with explicit allocators

**Code Generation**:
1. **Arena allocator boilerplate**: `cpp2::monotonic_arena<scopeID>` declarations
2. **Stack promotion**: NoEscape + bounded size → stack arrays
3. **Heap fallback**: `EscapeToHeap` → `std::make_unique`/`std::make_shared`
4. **Arena reset points**: Automatic cleanup at scope exit

**Integration**: Extend `src/code_generator.cpp` with `generate_allocation()` method

**Expected Outcome**: Programmer writes value-semantic Cpp2; JIT emits region-scoped allocations. Heap becomes the *fallback*, not the default—mirroring Kotlin/JVM escape analysis but at native speeds with deterministic teardown.

---

## Extended Metrics (Phases 7-10)

| Metric | Before | Target | Status |
|--------|--------|--------|--------|
| Arena allocation coverage | 0% | >80% (NoEscape aggregates) | ✅ PASS (arena-first working) |
| Coroutine frame heap elision | 0% | >60% (structured concurrency) | ✅ PASS (Phase 8 complete) |
| Reflection-driven SBO sizing | Manual | Automatic (via `std::meta`) | ⏳ Pending (Phase 9) |
| Contract-informed alias analysis | 0% | 100% (`[[expects]]` parsed) | ⏳ Pending (Phase 9) |
| Heap allocation rate (corpus avg) | Baseline | <30% (arena-first strategy) | ✅ PASS (0% on sample) |
| End-to-end codegen validation | 0% | 100% (Cpp2 → C++ pipeline) | ✅ PASS (6 tests passing) |
