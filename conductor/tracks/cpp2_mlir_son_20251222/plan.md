# Plan: Establish Core Cpp2 to MLIR Front-IR Conversion and Sea of Nodes Dialect Integration

## Phase 1: Foundational MLIR and FIR Dialect Setup

- [x] **Task:** Define the initial MLIR Dialect for the Front-IR (FIR).
    - [x] **Sub-task:** Write tests for the basic FIR dialect operations and types.
    - [x] **Sub-task:** Implement the basic FIR dialect operations and types.
- [x] **Task:** Implement the AST to FIR conversion for a basic "Hello, World" style function.
    - [x] **Sub-task:** Write tests for converting a simple function AST node to FIR.
    - [x] **Sub-task:** Implement the AST to FIR converter for simple functions.
- [ ] **Task:** Conductor - User Manual Verification 'Foundational MLIR and FIR Dialect Setup' (Protocol in workflow.md)

## Phase 2: Sea of Nodes (SON) Dialect and Lowering

- [ ] **Task:** Define the initial MLIR Dialect for the Sea of Nodes IR (`sond`).
    - [ ] **Sub-task:** Write tests for the core `sond` operations (e.g., constants, basic arithmetic).
    - [ ] **Sub-task:** Implement the core `sond` operations.
- [ ] **Task:** Implement the lowering from the FIR dialect to the `sond` dialect for simple functions.
    - [ ] **Sub-task:** Write tests for lowering a simple FIR function to `sond`.
    - [ ] **Sub-task:** Implement the FIR to `sond` lowering pass.
- [ ] **Task:** Conductor - User Manual Verification 'Sea of Nodes (SON) Dialect and Lowering' (Protocol in workflow.md)

## Phase 3: Pijul CRDT and Graph Serialization

- [ ] **Task:** Implement the core Pijul CRDT logic from first principles.
    - [ ] **Sub-task:** Write tests for the core CRDT functionalities (e.g., patch creation, application).
    - [ ] **Sub-task:** Implement the Pijul CRDT data structures and algorithms.
- [ ] **Task:** Implement serialization for the `sond` dialect using the Pijul CRDT implementation.
    - [ ] **Sub-task:** Write tests for serializing a simple `sond` graph.
    - [ ] **Sub-task:** Implement the `sond` serialization logic.
- [ ] **Task:** Implement deserialization for the `sond` dialect.
    - [ ] **Sub-task:** Write tests for deserializing a `sond` graph and verifying its integrity.
    - [ ] **Sub-task:** Implement the `sond` deserialization logic.
- [ ] **Task:** Conductor - User Manual Verification 'Pijul CRDT and Graph Serialization' (Protocol in workflow.md)
