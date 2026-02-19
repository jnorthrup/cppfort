# Regression Test Ranking for Inferred Cpp2 Constructs

To ensure the Sea-of-Nodes (SON) pipeline and MLIR dialect accurately reflect Cpp2 abstractions and prevent semantic drift, regression tests are ranked by their priority for safety analysis and semantic verification.

## Tier 1: Safety & Contracts (Critical for SON/Borrow Checker)
These tests validate the core value proposition of the MLIR pipeline: memory safety, lifecycle management, and contract enforcement.

### 1.1 Memory & Bounds Safety
*Focus: Bounds checking, null safety, pointer lifecycle.*
- `mixed-bounds-safety-with-assert-2.cpp`
- `mixed-bounds-safety-with-assert.cpp`
- `mixed-lifetime-safety-and-null-contracts.cpp`
- `mixed-lifetime-safety-pointer-init-4.cpp`
- `pure2-bounds-safety-span.cpp`
- `pure2-unsafe.cpp`

### 1.2 Initialization Safety
*Focus: Definite assignment, strict initialization rules.*
- `mixed-initialization-safety-3-contract-violation.cpp`
- `mixed-initialization-safety-3.cpp`
- `pure2-initialization-safety-with-else-if.cpp`

### 1.3 Contracts (Pre/Post/Assert)
*Focus: Runtime and static verification conditions.*
- `pure2-contracts.cpp`
- `pure2-assert-expected-not-null.cpp`
- `pure2-assert-optional-not-null.cpp`
- `pure2-assert-shared-ptr-not-null.cpp`
- `pure2-assert-unique-ptr-not-null.cpp`
- `mixed-captures-in-expressions-and-postconditions.cpp`

## Tier 2: Core Language Semantics (Critical for Isomorphic Loop)
These tests ensure the "C++ ↔ Cpp2" core loop preserves semantics for control flow, functions, and types.

### 2.1 Control Flow & Loops
*Focus: Loop structures (for, while, do), branching, and range-based iteration.*
- `mixed-intro-example-three-loops.cpp`
- `mixed-loops-out.cpp`
- `pure2-break-continue.cpp`
- `pure2-for-loop-range-with-lambda.cpp`
- `pure2-intro-example-three-loops.cpp`

### 2.2 Functions & Lambdas
*Focus: Function declaration, overloading, lambdas, and closures.*
- `mixed-function-expression-with-capture.cpp`
- `pure2-function-body-reflection.cpp`
- `pure2-function-multiple-forward-arguments.cpp`
- `pure2-function-single-expression-body-default-return.cpp`

### 2.3 Type System & Pattern Matching
*Focus: `inspect` expressions, type aliases, variants, and polymorphism.*
- `mixed-inspect-values.cpp`
- `pure2-types-basics.cpp`
- `pure2-types-inheritance.cpp`
- `pure2-is-with-polymorphic-types.cpp`
- `pure2-is-with-variable-and-value.cpp`

## Tier 3: Advanced Features (Specialized Semantics)
Tests for specific Cpp2 features that map to complex or unique MLIR constructs.

### 3.1 Unified Function Call Syntax (UFCS)
- `pure2-ufcs-member-access-and-chaining.cpp`
- `mixed-bugfix-for-ufcs-non-local.cpp`

### 3.2 Autodiff
- `mixed-autodiff-taylor.cpp`
- `pure2-autodiff.cpp`

### 3.3 Metaprogramming & Reflection
- `pure2-function-typeids.cpp`
- `pure2-types-ordering-via-meta-functions.cpp`

## Usage Guide

1.  **Semantic Drift Check**: Run Tier 1 and Tier 2 tests after any change to the `cppfort` emitter or `cpp2fir` lowering.
2.  **SON Validation**: Use Tier 1 tests to verify the efficacy of the borrow checker and safety analysis passes.
3.  **Feature Parity**: Use Tier 3 tests to track progress on supporting advanced Cpp2 features in the MLIR pipeline.
