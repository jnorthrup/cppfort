# Sea-of-Nodes Chapter TDD Track

## Track Objective
Implement one chapter of sea-of-nodes using Trikeshed composable hermetic abstractions as cpp2, with MLIR integration via TDD approach.

## Bounded Slice: Chapter 05 - Basic Arithmetic Operations

### Corpus Boundaries
- **Source**: `src/seaofnodes/chapter05/` (new directory)
- **Tests**: `tests/smoke/chapter05_test.cpp2`
- **MLIR**: `src/seaofnodes/chapter05/arith_ops.td`
- **Trikeshed**: Reuse existing patterns from `src/selfhost/trikeshed_*.cpp2`

### Implementation Order (TDD)
1. **Red**: Write failing test for basic arithmetic (add, sub, mul, div)
2. **Green**: Implement minimal cpp2 code to pass tests
3. **Refactor**: Apply Trikeshed hermetic abstractions
4. **MLIR**: Generate corresponding MLIR dialect operations
5. **Verify**: Validate cpp2 → MLIR mapping

### Dependencies
- Existing Trikeshed patterns (join, manifold, series, either)
- Cppfront transpilation pipeline
- MLIR Cpp2Dialect infrastructure
- Chapter 1-4 implementations as reference

### Verification Commands
```bash
# Build and test
ninja chapter05_test
./tests/smoke/chapter05_test

# Verify MLIR generation
mlir-opt --cpp2-dialect chapter05.mlir
```

## Slave Assignment
- **Worker A**: TDD implementation of arithmetic operations in cpp2
- **Worker B**: MLIR dialect generation and validation

## Acceptance Criteria
- All tests pass with 100% coverage
- Hermetic abstractions properly applied
- MLIR generation verified
- No code outside repo boundaries