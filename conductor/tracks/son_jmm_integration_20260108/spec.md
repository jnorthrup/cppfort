# Java Memory Model Integration for Cpp2 SON Dialect

**Track ID:** `son_jmm_integration_20260108`
**Date:** 2026-01-08
**Type:** Feature (Mandatory)

## Overview

**MANDATORY IMPLEMENTATION:** Java Memory Model (JMM) guarantees must be implemented in Cpp2 through MLIR dispatch-level analysis that informs the emitter how to marshal memory operations based on target capabilities.

### Architecture Position

```
Cpp2 Source → Transpiler → SON Dialect → Dispatch Analysis → Emitter → Target Code
                                              ↓
                                 JMM Requirements
                                     (Target-Aware)
```

**Key Insight:** JMM is NOT a single implementation point - it's **dispatch analysis** that determines:
1. What memory order is required (happens-before, SC, release/acquire)
2. What the target supports (C++20 std::atomic, hardware fences, etc.)
3. How to marshal the operation (emit std::atomic, fence, or optimal target-specific)

## JMM Guarantees (All Required)

### 1. Happens-Before Relationships
- Formal partial ordering for SON operations
- Edges between SON nodes establishing memory visibility
- Transitive closure for dependency chains
- Maps to: `memory_order_acquire`, `memory_order_release`, `memory_order_relaxed`

### 2. Volatile Semantics (Sequential Consistency)
- SON operations marked with SC requirements
- Total order on volatile reads/writes
- Fences for synchronization points
- Maps to: `memory_order_seq_cst`

### 3. Final Field Safety
- Freeze semantics for constructor-exited entities
- Safe publication guarantees through SON
- Escape analysis integration
- Maps to: `atomic_thread_fence(memory_order_release)` after constructor

### 4. Memory Visibility Guarantees
- Thread-local vs shared entity tracking
- Cache invalidation semantics in SON
- Publication/subscription patterns
- Maps to: `std::atomic<T>` for shared variables

## Functional Requirements (Mandatory)

| FR | Description | Priority |
|----|-------------|----------|
| **FR-1** | SON nodes annotated with JMM memory requirements | MANDATORY |
| **FR-2** | Dispatch pass analyzes target capabilities | MANDATORY |
| **FR-3** | Emitter marshals based on dispatch decisions | MANDATORY |
| **FR-4** | Happens-before tracking → memory_order selection | MANDATORY |
| **FR-5** | Volatile SC → sequential consistency emission | MANDATORY |
| **FR-6** | Final field freeze → constructor fences | MANDATORY |
| **FR-7** | Thread-safe entity lifecycle | MANDATORY |

## Dispatch Decision Matrix

| JMM Requirement | C++20 Target | Alternative Target |
|-----------------|--------------|-------------------|
| Happens-before | `memory_order_acquire/release` | Target-specific |
| Volatile read/write | `memory_order_seq_cst` | Target SC primitive |
| Final field | `atomic_thread_fence(release)` | Target fence |
| Visibility | `std::atomic<T>` | Target atomic type |

## Implementation Components

### 1. SON Dialect Extensions
- Memory order attributes on operations
- JMM constraint metadata
- Happens-before edge tracking

### 2. Dispatch Analysis Pass
- Target capability queries
- Memory order selection
- Marshaling strategy determination

### 3. Emitter Integration
- Target-specific code generation
- Fences/barriers emission
- Atomic type marshaling

## Acceptance Criteria

- [ ] SON operations carry JMM requirement metadata
- [ ] Dispatch pass maps requirements to target capabilities
- [ ] Emitter generates correct C++20 std::atomic code
- [ ] Happens-before edges produce correct memory_order
- [ ] Volatile emits seq_cst operations
- [ ] Final fields get constructor fences
- [ ] Tests verify JMM guarantees in emitted code

## Out of Scope

- JMM implementation in C++ runtime (assumes std::atomic exists)
- Hardware-specific memory models (ARM/IA64 differences - handled by C++ compiler)
- Java interop/transpilation (focus on Cpp2 semantics only)
- Alternative target backends (beyond C++20)

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| JMM attribute coverage | 100% | All concurrent ops annotated |
| Dispatch accuracy | 100% | Correct memory orders emitted |
| Test coverage | >80% | Litmus + unit tests |
| Performance | <5% overhead | vs hand-written std::atomic |

## References

- JSR-133: Java Memory Model and Thread Specification
- C++20 Standard: Chapter 31 (Atomic operations)
- LLVM Memory Model: https://llvm.org/docs/MemoryModel.html
- Cpp2 SON Dialect: `include/Cpp2SON.td`
