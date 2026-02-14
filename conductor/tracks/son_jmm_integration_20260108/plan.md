# Implementation Plan: Java Memory Model Integration for Cpp2 SON Dialect

## Phase 1: SON Dialect JMM Extensions

- [x] Define memory order attributes in Cpp2SON dialect
  - [x] Add `jmm_happens_before` attribute to Cpp2SON.td
  - [x] Add `jmm_volatile` attribute (SC semantics)
  - [x] Add `jmm_final_field` attribute (freeze semantics)
  - [x] Add `jmm_visibility` attribute (thread-local vs shared)
- [x] Extend SON operations with JMM metadata support
  - [x] Attach memory requirements to load/store ops
  - [x] Annotate concurrent operations (send, recv, spawn, await)
  - [x] Mark constructor boundaries for final fields
  - [x] Add thread-local vs shared variable tracking
- [x] Create JMM constraint verification in SON
  - [x] Validate happens-before edge consistency (DFS cycle detection in Cpp2SONJMMVerification.cpp:322-368)
  - [x] Check volatile operation total ordering (volatile requires shared visibility in :188-198, :217-226)
  - [x] Verify final field freeze timing (final field frozen check in :228-236)
  - [x] Detect unsafe publication patterns (constructor visibility check in :253-267)
  - NOTE: Implementation complete, testing blocked by SON dialect disabled in build (LLVM 21 FieldParser issue)
- [x] Write tests for SON JMM attributes
  - [~] Test attribute parsing from Cpp2 source - BLOCKED (requires Phase 2 parser integration)
  - [x] Test attribute attachment to SON ops (test_jmm_attribute_attachment_to_ops)
  - [x] Test constraint validation rules (test_jmm_constraint_validation)
  - [~] Test MLIR round-trip with JMM ops - BLOCKED (requires operational SON dialect build)

## Phase 2: Dispatch Analysis Pass

- [ ] Implement target capability query interface
  - [ ] Define `TargetCapabilities` abstract interface
  - [ ] Implement C++20 target capability descriptor
  - [ ] Query: supported memory orders (relaxed, acquire, release, seq_cst)
  - [ ] Query: supported atomic types (std::atomic, fences)
  - [ ] Query: hardware fence support
- [ ] Create JMM dispatch analysis pass
  - [ ] Analyze SON graph for JMM requirements
  - [ ] Build happens-before dependency graph
  - [ ] Match requirements to target capabilities
  - [ ] Generate marshaling decisions for each operation
  - [ ] Output dispatch metadata for emitter
- [ ] Implement memory order selection logic
  - [ ] Happens-before → acquire/release mapping
  - [ ] Write → release, Read → acquire
  - [ ] Volatile → seq_cst mapping
  - [ ] Final field → release fence after constructor
  - [ ] Relax safe operations to memory_order_relaxed
- [ ] Write tests for dispatch analysis
  - [ ] Test target capability queries
  - [ ] Test memory order selection algorithms
  - [ ] Test happens-before graph construction
  - [ ] Test marshaling decision generation
  - [ ] Test edge cases (cyclic dependencies, missing edges)

## Phase 3: Emitter Integration

- [ ] Extend code generator for JMM emission
  - [ ] Emit `std::atomic<T>` for shared variables
  - [ ] Emit `T` for thread-local variables
  - [ ] Emit memory_order parameters on load/store
  - [ ] Emit `std::atomic_thread_fence` for barriers
  - [ ] Add necessary includes (<atomic>)
- [ ] Implement volatile operation emission
  - [ ] Generate seq_cst loads for volatile reads
  - [ ] Generate seq_cst stores for volatile writes
  - [ ] Add total order verification (debug/assert mode)
- [ ] Implement final field emission
  - [ ] Emit release fence after constructor completion
  - [ ] Ensure object visibility before publication
  - [ ] Integrate with escape analysis for freeze points
- [ ] Write end-to-end JMM emission tests
  - [ ] Compile Cpp2 source with JMM annotations
  - [ ] Verify emitted C++ has correct std::atomic
  - [ ] Test happens-before in emitted code
  - [ ] Test volatile SC semantics
  - [ ] Test final field safety patterns
  - [ ] Test multi-threaded scenarios (2+ threads)

## Phase 4: Validation and Documentation

- [ ] Create comprehensive JMM test suite
  - [ ] Happens-before chain tests (transitive closure)
  - [ ] Volatile visibility tests (across threads)
  - [ ] Final field publication tests (safe construct)
  - [ ] Data race detection tests (should fail without JMM)
  - [ ] Litmus tests for concurrency scenarios
- [ ] Verify emitted code against JMM spec
  - [ ] Formal verification of memory orders
  - [ ] Compare against Java reference behavior
  - [ ] Test on actual hardware (x86_64, ARM64 if available)
  - [ ] ThreadSanitizer verification
- [ ] Document JMM implementation in Cpp2
  - [ ] JMM guarantees provided by Cpp2
  - [ ] Mapping from JMM concepts to C++20
  - [ ] Usage examples and patterns
  - [ ] Performance characteristics
  - [ ] Migration guide from Java JMM
- [ ] Write performance benchmarks
  - [ ] Atomic operation overhead vs plain accesses
  - [ ] Fence cost analysis
  - [ ] Optimization opportunities (relaxation)
  - [ ] Comparison to hand-written std::atomic

## Phase 5: Integration Testing

- [ ] Integration with existing Cpp2 features
  - [ ] Channel operations with JMM semantics
  - [ ] Coroutine frame elision with memory ordering
  - [ ] Arena allocation with final field safety
  - [ ] Escape analysis integration
- [ ] Regression testing with JMM enabled
  - [ ] Run existing corpus with JMM annotations
  - [ ] Verify no functional regressions
  - [ ] Check performance impact
- [ ] Concurrency safety validation
  - [ ] Verify all concurrent ops are JMM-compliant
  - [ ] Test data race detection
  - [ ] Validate lock-free patterns

## Checkpoint Tasks

- [ ] Task: Conductor - User Manual Verification 'Phase 1: SON Dialect JMM Extensions' (Protocol in workflow.md)
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Dispatch Analysis Pass' (Protocol in workflow.md)
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Emitter Integration' (Protocol in workflow.md)
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Validation and Documentation' (Protocol in workflow.md)
- [ ] Task: Conductor - User Manual Verification 'Phase 5: Integration Testing' (Protocol in workflow.md)
