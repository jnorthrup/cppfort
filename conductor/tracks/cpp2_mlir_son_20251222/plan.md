# Plan: Establish Core Cpp2 to MLIR Front-IR Conversion and Sea of Nodes Dialect Integration

## Phase 1: Foundational MLIR and FIR Dialect Setup [checkpoint: 89ca649]

- [x] **Task:** Define the initial MLIR Dialect for the Front-IR (FIR).
    - [x] **Sub-task:** Write tests for the basic FIR dialect operations and types.
    - [x] **Sub-task:** Implement the basic FIR dialect operations and types.
- [x] **Task:** Implement the AST to FIR conversion for a basic "Hello, World" style function.
    - [x] **Sub-task:** Write tests for converting a simple function AST node to FIR.
    - [x] **Sub-task:** Implement the AST to FIR converter for simple functions.
- [x] **Task:** Clang AST corpus processing for cppfront regression tests.
    - [x] **Sub-task:** Process 146 cppfront regression tests (cpp2 → C++1 → Clang AST).
    - [x] **Sub-task:** Extract 40 function signatures with parameter qualifiers.
    - [x] **Sub-task:** Integrate corpus-derived patterns into AST definition:
        - ParameterQualifier enum (InOut, Out, Move, Forward, Virtual, Override)
        - Qualifiers on FunctionDeclaration::Parameter, LambdaExpression::Parameter, VariableDeclaration
        - UFCS tracking (is_ufcs flag on CallExpression)
        - Bounds checking (has_bounds_check on SubscriptExpression, BoundsCheckExpression)
- [ ] **Task:** Complete semantic mapping from corpus patterns.
    - [ ] **Sub-task:** Parse Cpp2 qualifiers (inout, out, move, forward, virtual, override).
    - [ ] **Sub-task:** Convert Clang AST patterns to cpp2 AST nodes using corpus mappings.
    - [ ] **Sub-task:** AST→FIR conversion using corpus-derived semantics.
    - [ ] **Sub-task:** MLIR ops tagged with corpus semantics.
- [ ] **Task:** Conductor - User Manual Verification 'Foundational MLIR and FIR Dialect Setup' (Protocol in workflow.md)

## Phase 2: Sea of Nodes (SON) Dialect and Lowering [checkpoint: d5b5758]

- [x] **Task:** Define the initial MLIR Dialect for the Sea of Nodes IR (`sond`).
    - [x] **Sub-task:** Write tests for the core `sond` operations (e.g., constants, basic arithmetic).
    - [x] **Sub-task:** Implement the core `sond` operations.
- [x] **Task:** Implement the lowering from the FIR dialect to the `sond` dialect for simple functions.
    - [x] **Sub-task:** Write tests for lowering a simple FIR function to `sond`.
    - [x] **Sub-task:** Implement the FIR to `sond` lowering pass.
- [x] **Task:** Conductor - User Manual Verification 'Sea of Nodes (SON) Dialect and Lowering' (Protocol in workflow.md)

## Phase 3: Pijul CRDT and Graph Serialization [checkpoint: 9cd579f]

- [x] **Task:** Implement the core Pijul CRDT logic from first principles.
    - [x] **Sub-task:** Write tests for the core CRDT functionalities (e.g., patch creation, application).
    - [x] **Sub-task:** Implement the Pijul CRDT data structures and algorithms.
- [x] **Task:** Implement serialization for the `sond` dialect using the Pijul CRDT implementation.
    - [x] **Sub-task:** Write tests for serializing a simple `sond` graph.
    - [x] **Sub-task:** Implement the `sond` serialization logic.
- [x] **Task:** Implement deserialization for the `sond` dialect.
    - [x] **Sub-task:** Write tests for deserializing a `sond` graph and verifying its integrity.
    - [x] **Sub-task:** Implement the `sond` deserialization logic.
- [x] **Task:** Conductor - User Manual Verification 'Pijul CRDT and Graph Serialization' (Protocol in workflow.md)
