# Project Objectives

## 1. Isomorphic C++ Transpilation
The primary immediate goal is to provide a robust **Cpp2 → C++ transpiler** (`cppfort`) that produces output semantically identical to `cppfront`.
- **Goal**: Clean `wdiff` and whitespace compatibility with `cppfront` output.
- **Purpose**: Serve as a host/testing ground for Cpp2 corpus and ensuring semantic correctness.
- **Future**: Potential to serve as an indenter/cleaner for Cpp2.

## 2. Roundtrip Transpilation
The project aims to support **roundtrip capability** with a specific flow: **Cpp2 (or best C++) → MLIR → Cpp2/C++**.
- **Constraint**: The transformation direction is strictly from the "best" available source (Cpp2 or clean C++) *into* MLIR.
- **Goal**: Use the graph representation (SoN) as a unified source of truth for optimization, safety analysis, and code generation.
- **Roundtrip**: The "roundtrip" implies generating code back from this MLIR, minimizing n-way mapping complexity by using MLIR as the central hub.

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
