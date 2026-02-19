# Project Objectives

## 1. Isomorphic C++ Transpilation
The primary immediate goal is to provide a robust **Cpp2 → C++ transpiler** (`cppfort`) that produces output semantically identical to `cppfront`.
- **Goal**: Clean `wdiff` and whitespace compatibility with `cppfront` output.
- **Purpose**: Serve as a host/testing ground for Cpp2 corpus and ensuring semantic correctness.
- **Future**: Potential to serve as an indenter/cleaner for Cpp2.

## 2. Roundtrip Transpilation
The project supports **roundtrip capability** with a specific validation scope.
- **Flow**: Cpp2 (or best C++) → MLIR → Cpp2/C++.
- **Validation**: Roundtrip validation is strictly defined as **C++ ↔ C++**.
- **Constraint**: The system validates its correctness by ensuring that C++ input can be processed (e.g., through the SON pipeline for safety/optimization) and emitted back as C++ with verified fidelity.
- **Non-Goal**: Reverse mapping from arbitrary C++ to MLIR is not a goal unless it supports this specific roundtrip for SON benefits.

## 3. Safety Analysis via Sea-of-Nodes (SON)
The MLIR-based Sea-of-Nodes (SON) pipeline is being developed specifically to implement **memory and lifecycle safety checks** (a "borrow checker" for Cpp2).
- **Dialects**:
    - `cpp2fir`: Frontend IR representing Cpp2 constructs.
    - `sond` (Sea-of-Nodes Dialect): Analysis IR.
- **Focus**:
    - Lifecycle safety (use-after-move, use-after-free).
    - Memory safety (bounds checks, null checks).
    - Concurrency safety (via JMM/Happens-Before modeling in `sond`).

## 4. Semantic Grounding via Clang
Leverage **Clang AST** to reverse-map semantic features of the C++ language to the Cpp2/Cppfort variant.
- **Goal**: Extend and verify the semantic foundations of Cpp2 by cross-referencing with standard C++ semantics extracted via Clang.

## 5. Future Directions
- **C Generation**: Transpilation to C is a secondary, lower-priority target.
- **Module Structure**: C/C++ module structure alignment.
