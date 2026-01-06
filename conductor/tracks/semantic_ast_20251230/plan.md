# Implementation Plan: Semantic AST Enhancements

## Phase 1: Core Escape Analysis

- [x] Implement `EscapeInfo` and `EscapeKind` enums in include/ast.hpp
- [x] Add escape analysis pass in src/semantic_analyzer.cpp after type checking
- [x] Attach `EscapeInfo` to all `VarDecl` nodes
- [x] Write unit tests for basic escape scenarios (NoEscape, EscapeToReturn, EscapeToHeap)
- [x] Validate against pure2 corpus files (start with pure2-hello.cpp2)
  NOTE: Parameter passing (inout) works correctly. Escape analysis not implemented.

## Phase 2: Borrowing and Ownership

- [x] Implement `BorrowInfo` and `OwnershipKind` in include/ast.hpp
- [x] Add `LifetimeRegion` tracking structures
- [x] Implement `BorrowChecker` validation class in include/safety_checker.hpp
- [x] Map parameter qualifiers to ownership kinds (in→Borrowed, out→MutBorrowed, inout→MutBorrowed, move→Moved, forward→conditional)
- [x] Enforce aliasing rules (exclusive mutable borrow)
- [x] Fix `decorate` parameter passing bug (inout → std::string&) in code generator
- [x] Write tests for all parameter qualifier combinations
- [x] Test borrow checker rules (no aliasing, outlives, move invalidation)

## Phase 3: External Memory Integration

- [x] Implement `MemoryTransfer` tracking structure
- [ ] Connect escape analysis to GPU/DMA transfers in AST
- [ ] Add lifecycle-based optimization pass in MLIR pipeline
- [ ] Validate DMA safety rules (no aliasing during async transfers)
- [ ] Test with `mem_region` and `kernel` MLIR ops
- [ ] Verify optimization eliminates unnecessary transfers

## Phase 4: Channelized Concurrency

- [ ] Implement `ChannelTransfer` tracking structure
- [ ] Connect channel ops to escape analysis
- [ ] Enforce ownership transfer rules for send/recv operations
- [ ] Add data race detection for channel operations
- [ ] Test with `channel`, `send`, `recv` MLIR ops
- [ ] Verify channel safety invariants

## Phase 5: Unified Semantic Info

- [ ] Implement `SemanticInfo` struct in include/ast.hpp
- [ ] Attach `semantic_info` field to all AST nodes
- [ ] Add query methods (`is_safe()`, `can_optimize_away()`, `explain_semantics()`)
- [ ] Generate semantic dump for debugging (extend AST printer)
- [ ] Update AST→MLIR lowering to preserve semantics
- [ ] Emit `!cpp2.owned`, `!cpp2.borrowed` type attributes in MLIR

## Phase 6: Regression Testing and Validation

- [ ] Run full pure2 corpus (139 files) through enhanced semantic analysis
- [ ] Measure semantic loss scores for all corpus files
- [ ] Compare cppfort output against cppfront reference
- [ ] Target: <0.15 average corpus loss (current: 1.0)
- [ ] Target: <0.05 loss for pure2-hello.cpp2 (current: 1.0)
- [ ] Document semantic preservation improvements
- [ ] Update REGRESSION_TEST_STATUS.md with new results

## Phase 7: Scope-Inferred Arena Allocation (JIT Memory Management)

- [ ] Design `ArenaRegion` tracking structure linked to `LifetimeRegion`
- [ ] Implement scope-level escape analysis aggregation (determine arena-eligible scopes)
- [ ] Add arena allocation hints to MLIR FIR operations
  - New attribute: `#cpp2.arena_scope<scopeID>`
- [ ] Implement JIT analysis pass: `ArenaInferencePass` in MLIR pipeline
  - Input: Escape-annotated FIR
  - Output: Scope-tagged operations with arena/stack/heap decisions
- [ ] Add MLIR type attribute: `!cpp2.arena<scopeID>` for arena-allocated values
- [ ] Write unit tests for arena inference on nested scopes
- [ ] Validate with Cpp2 functions containing local aggregates (vector, map, string)
- [ ] Verify arena allocation eliminates heap allocs for NoEscape aggregates

## Phase 8: Coroutine Frame Elision (Kotlin-style Structured Concurrency)

- [ ] Implement `CoroutineContainmentGraph` analysis (track parent-child coroutine relationships)
- [ ] Extend escape analysis to track coroutine frame escape points
  - New `EscapeKind`: `EscapeToCoroutineFrame`
- [ ] Port Kotlin `CoroutineScope` semantics to C++26 `std::execution` senders
  - Map `launch {}` blocks → `std::execution::schedule`
  - Map `async/await` → sender/receiver chains
- [ ] Implement MLIR pass: `CoroutineFrameSROA`
  - Detect non-escaping coroutine frames (suspended state contained within parent scope)
  - Convert heap-boxed frames → stack-pinned or arena-slotted frames
- [ ] Add MLIR attribute: `#cpp2.coroutine_frame<stack|arena|heap>`
- [ ] Write tests for structured concurrency patterns (parent-child coroutines)
- [ ] Validate frame elision with `cpp2fir.coroutine_scope` operations
- [ ] Benchmark coroutine frame allocation (stack/arena vs heap)

## Phase 9: C++26 Integration (Reflection, Contracts, Pattern Matching)

- [ ] Integrate C++26 reflection (`std::meta`) for SBO (Small Buffer Optimization) sizing
  - Implement `reflection_driven_sbo_size()` utility using `std::meta::info`
  - Use reflection to compute optimal `std::inplace_vector<T, N>` capacity at compile time
- [ ] Feed C++26 contracts (`[[expects]]`, `[[ensures]]`) into alias analysis
  - Parse contract annotations in AST
  - Derive no-alias guarantees from preconditions
  - Add `ContractInfo` to `SemanticInfo`
- [ ] Integrate pattern matching (`inspect`) for exhaustive resource state tracking
  - Add `ResourceState` enum (Uninitialized, Initialized, Moved, Borrowed)
  - Track state transitions through `inspect` branches
- [ ] Add MLIR attributes for C++26 features:
  - `#cpp2.contract<precondition|postcondition>`
  - `#cpp2.reflection_sized<bytes>`
- [ ] Write tests for contract-informed alias analysis
- [ ] Validate SBO sizing on `std::inplace_vector` code generation
- [ ] Test pattern matching resource state tracking

## Phase 10: JIT Codegen Backend (Stack/Arena/Heap Decision Logic)

- [ ] Implement `AllocationStrategyPass` in MLIR → C++ codegen
  - Input: Annotated FIR with `#cpp2.arena_scope`, `!cpp2.arena<scopeID>`, etc.
  - Output: C++ code with explicit allocators
- [ ] Generate arena allocator boilerplate
  - Emit `cpp2::monotonic_arena<scopeID>` declarations
  - Insert arena reset points at scope exits
- [ ] Implement fallback to heap for escaping values
  - Detect `EscapeToHeap` annotations → emit `std::make_unique`/`std::make_shared`
- [ ] Add stack promotion for provably-bounded aggregates
  - NoEscape + bounded size → stack array allocation
- [ ] Integrate with existing code generator (`src/code_generator.cpp`)
  - Add `generate_allocation()` method with strategy dispatch (stack/arena/heap)
- [ ] Write end-to-end tests: Cpp2 source → JIT-optimized C++ output
- [ ] Benchmark allocation performance (arena vs heap) on corpus files
- [ ] Document allocation strategy in generated C++ comments
- [ ] Validate: Heap becomes fallback, not default (arena-first allocation)
