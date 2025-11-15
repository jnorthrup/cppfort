# C→C++ Inheritance Plan

This document specifies how the project merges the majority of C constructs into the C++ emission path, treating differences as specialized branches behind a `grammar_mode == C` flag. It complements `tests/regression-tests/SCANNER_RECONSTRUCTION_COMPLETE.md`, which previously focused only on CPP2→C++.

## Goals
- Make C a first-class source grammar in the unified orbit system.
- Prefer unified patterns that emit identical or trivially-adapted C++.
- Isolate C-only feature handling behind a small set of specializations.
- Keep the midpoint small with a pseudo C doc‑dialect used only for tests/docs.

## Mapping Overview

### Unify Now (Identity or Trivial Differences)
- Declarations (functions, globals)
- Expressions (arith/logic/comparisons)
- Control flow (if/while/for/switch)
- Pointers and address-of/deref
- Struct/enum/union definitions
- Arrays and indexing
- Function pointers

### Specialize (C surface semantics differ)
- Designated initializers (C99) → Prefer C++20 aggregate init; else rewrite
- VLAs (C99) → Rewrite to `std::vector` and explicit bounds
- Compound literals (C99) → Introduce temporaries/aggregate init in C++
- Flexible array members → Trailing storage field pattern
- `restrict`, `_Atomic`, `_Generic` → Attributes/`std::atomic`/overload sets

## Pseudo C Doc‑Dialect (Midpoint Naming)
This is a documentation/testing vocabulary to align with the `cpp2.*` sections. It does not introduce a new runtime dialect; actual lowering uses existing MLIR dialects or direct C++ emission.

- `c.func`        → MLIR `func.func` or C++ function
- `c.call`        → MLIR `func.call`
- `c.ret`         → MLIR `func.return`
- `c.alloca`      → MLIR `llvm.alloca` or C++ stack local
- `c.load`        → MLIR `memref/llvm.load` or `*p`
- `c.store`       → MLIR `memref/llvm.store` or `*p = v`
- `c.ptr.add`     → pointer arithmetic
- `c.struct.decl` → type annotation for emission

## Validation

Extend the existing validation to include C→C++ inheritance checks:

```cpp
struct GapValidationC : GapValidation {
    bool c_equivalent_compiles;  // C-derived C++ compiles
    bool no_vla_left;            // VLAs rewritten
    bool c_struct_layout_kept;   // Struct ABI layout preserved
};
```

Matrix excerpt (see full table in SCANNER doc):

| C pattern        | unified? | needs mode | c_equiv_compiles | layout_kept |
|------------------|----------|------------|------------------|-------------|
| function_decl    | ✅       | ❌         | ✅               | n/a         |
| pointer_deref    | ✅       | ❌         | ✅               | n/a         |
| struct_def       | ✅       | ❌         | ✅               | ✅          |
| designated_init  | ✅       | ✅         | ✅               | ✅          |
| vla_use          | ❌→rewrite| ✅        | ✅               | n/a         |

## Integration Points
- Orbit categories: `src/stage0/grammar_tree.*` already include C; classifiers can set branch label `C`.
- Projection: `src/stage0/projection_oracle.h` includes `C`/`CPP` alongside MLIR; roundtrip boost rules should treat C↔C++ similarly to CPP2↔C++ for overlapping patterns.
- Patterns: `src/stage0/unified_orbit_patterns.*` carry mode bits; extend to set `grammar_mode = C` where C-specific rewriting is required.

## Next Steps
- Document designated initializer rewrite templates with examples.
- Add a smoke test that validates `c_equivalent_compiles` for a small corpus of C files emitted as C++.
- Expand struct layout validation using `static_assert(sizeof(...))` on representative cases.

