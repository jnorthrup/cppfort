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

## Implementation Roadmap

### Phase 1: Core Escape Analysis (2-3 days)
- [ ] Implement `EscapeInfo` and `EscapeKind` enums
- [ ] Add escape analysis pass after type checking
- [ ] Attach `EscapeInfo` to all `VarDecl` nodes
- [ ] Write tests for basic escape scenarios
- [ ] Validate against pure2 corpus files

### Phase 2: Borrowing and Ownership (3-4 days)
- [ ] Implement `BorrowInfo` and `OwnershipKind`
- [ ] Add `LifetimeRegion` tracking
- [ ] Implement `BorrowChecker` validation
- [ ] Map parameter qualifiers to ownership kinds
- [ ] Enforce aliasing rules
- [ ] Fix `decorate` parameter passing bug (inout → std::string&)

### Phase 3: External Memory Integration (2-3 days)
- [ ] Implement `MemoryTransfer` tracking
- [ ] Connect escape analysis to GPU/DMA transfers
- [ ] Add lifecycle-based optimization pass
- [ ] Validate DMA safety rules
- [ ] Test with `mem_region` and `kernel` ops

### Phase 4: Channelized Concurrency (2-3 days)
- [ ] Implement `ChannelTransfer` tracking
- [ ] Connect channel ops to escape analysis
- [ ] Enforce ownership transfer rules for send/recv
- [ ] Add data race detection
- [ ] Test with `channel`, `send`, `recv` ops

### Phase 5: Unified Semantic Info (2 days)
- [ ] Implement `SemanticInfo` struct
- [ ] Attach to all AST nodes
- [ ] Add query methods (`is_safe()`, `can_optimize_away()`)
- [ ] Generate semantic dump for debugging
- [ ] Update AST→MLIR lowering to preserve semantics

### Phase 6: Regression Testing and Validation (3 days)
- [ ] Run full pure2 corpus (139 files) through enhanced analysis
- [ ] Measure semantic loss scores
- [ ] Compare against cppfront reference
- [ ] Target: <0.15 average corpus loss
- [ ] Document semantic preservation improvements

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

### Documentation
- Update `docs/AST_MAPPING_STATUS.md` with semantic enhancements
- Create `docs/SEMANTIC_ANALYSIS.md` explaining analysis passes
- Document parameter qualifier semantics in `docs/PARAMETER_QUALIFIERS.md`

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

**Next Steps**: Begin Phase 1 (Core Escape Analysis) implementation after SCCP track closure.
