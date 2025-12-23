# Spec: Establish Core Cpp2 to MLIR Front-IR Conversion and Sea of Nodes Dialect Integration

## 1. Overview

This track focuses on building the foundational infrastructure for the transpiler's core transformation pipeline. It involves creating a robust bridge between the Cpp2 language frontend and the MLIR intermediate representation (IR), and establishing the initial integration point for the Sea of Nodes (SON) compiler backend through a dedicated MLIR dialect.

## 2. Key Features & Requirements

### 2.1 Cpp2 to MLIR Front-IR Conversion
- **AST to FIR Bridge:** Develop a component that traverses the Cpp2 Abstract Syntax Tree (AST) and converts it into a Front-IR (FIR) based on MLIR.
- **FIR Dialect:** Define an MLIR dialect that faithfully represents all Cpp2 language constructs, including types, functions, unified function calls, and contracts.
- **Type System Mapping:** Ensure a one-to-one mapping between Cpp2's type system and the MLIR type system within the FIR dialect.
- **Error Handling:** Implement robust error reporting for conversion failures between AST and FIR.

### 2.2 Sea of Nodes Dialect Integration
- **SON Dialect Definition:** Define a new MLIR dialect (`sond`) that represents the core concepts of the Sea of Nodes IR graph, including nodes for constants, operations, and control flow.
- **FIR to SON Lowering:** Implement a conversion pass that lowers the FIR dialect to the `sond` dialect. This will involve mapping high-level Cpp2 constructs to their lower-level graph representations.
- **Dialect Verification:** The `sond` dialect must include verifiers to ensure the integrity and correctness of the SON graph representation.

### 2.3 Pijul CRDT Integration for Graph Serialization
- **Inline CRDT Implementation:** Develop the core Pijul CRDT logic inline, without external dependencies, as per the tech stack guidelines.
- **Graph SerDe:** Implement serialization and deserialization (SerDe) for the `sond` dialect's graph representation using the custom Pijul CRDT implementation. This will be used for caching and future JIT injection.

## 3. Out of Scope
- **Full Cpp2 Language Support:** This track will focus on a subset of Cpp2 features to establish the pipeline. Full language support will be handled in subsequent tracks.
- **Optimization:** No significant optimizations will be implemented in this track. The focus is on correctness and establishing the IR pipeline.
- **Code Generation from SON:** This track will not generate C++ or machine code from the SON dialect. It will only establish the dialect and the lowering from FIR.
- **JIT Injection:** While the groundwork for serialization will be laid, the actual fractal JIT injection is out of scope for this track.

## 4. Acceptance Criteria
- A Cpp2 source file with a representative subset of features can be successfully converted into an MLIR file containing the FIR dialect.
- The generated FIR dialect can be successfully lowered into the `sond` dialect without errors.
- The `sond` dialect representation can be serialized to a binary format using the Pijul CRDT implementation and then deserialized back into a valid `sond` MLIR representation.
- Unit tests cover the AST to FIR conversion, FIR to `sond` lowering, and SerDe process, achieving at least 20% code coverage.
