# Implementation Plan: Semantic AST Enhancements

## Phase 1: Core Escape Analysis

- [x] Implement `EscapeInfo` and `EscapeKind` enums in include/ast.hpp
- [x] Add escape analysis pass in src/semantic_analyzer.cpp after type checking
- [x] Attach `EscapeInfo` to all `VarDecl` nodes
- [ ] Write unit tests for basic escape scenarios (NoEscape, EscapeToReturn, EscapeToHeap)
- [ ] Validate against pure2 corpus files (start with pure2-hello.cpp2)

## Phase 2: Borrowing and Ownership

- [ ] Implement `BorrowInfo` and `OwnershipKind` in include/ast.hpp
- [ ] Add `LifetimeRegion` tracking structures
- [ ] Implement `BorrowChecker` validation class in include/safety_checker.hpp
- [ ] Map parameter qualifiers to ownership kinds (in→Borrowed, out→MutBorrowed, inout→MutBorrowed, move→Moved, forward→conditional)
- [ ] Enforce aliasing rules (exclusive mutable borrow)
- [ ] Fix `decorate` parameter passing bug (inout → std::string&) in code generator
- [ ] Write tests for all parameter qualifier combinations
- [ ] Test borrow checker rules (no aliasing, outlives, move invalidation)

## Phase 3: External Memory Integration

- [ ] Implement `MemoryTransfer` tracking structure
- [ ] Connect escape analysis to GPU/DMA transfers in AST
- [ ] Add lifecycle-based optimization pass in MLIR pipeline
- [ ] Validate DMA safety rules (no aliasing during async transfers)
- [ ] Test with `mem_region` and `kernel` MLIR ops
- [ ] Verify optimization eliminates unnecessary transfers

## Phase 4: Channelized Concurrency

- [ ] Implement `ChannelTransfer` tracking structure
- [ ] Connect channel ops to escape analysis
- [ ] Enforce ownership transfer rules for send/recv operations
- [ ] Add data race detection for channel operations
- [ ] Test with `channel`, `send`, `recv` MLIR ops
- [ ] Verify channel safety invariants

## Phase 5: Unified Semantic Info

- [ ] Implement `SemanticInfo` struct in include/ast.hpp
- [ ] Attach `semantic_info` field to all AST nodes
- [ ] Add query methods (`is_safe()`, `can_optimize_away()`, `explain_semantics()`)
- [ ] Generate semantic dump for debugging (extend AST printer)
- [ ] Update AST→MLIR lowering to preserve semantics
- [ ] Emit `!cpp2.owned`, `!cpp2.borrowed` type attributes in MLIR

## Phase 6: Regression Testing and Validation

- [ ] Run full pure2 corpus (139 files) through enhanced semantic analysis
- [ ] Measure semantic loss scores for all corpus files
- [ ] Compare cppfort output against cppfront reference
- [ ] Target: <0.15 average corpus loss (current: 1.0)
- [ ] Target: <0.05 loss for pure2-hello.cpp2 (current: 1.0)
- [ ] Document semantic preservation improvements
- [ ] Update REGRESSION_TEST_STATUS.md with new results
