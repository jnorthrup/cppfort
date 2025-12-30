# Plan: Regression Tests Validation Framework

**Track**: regression_tests_validation_20251230
**Created**: 2025-12-30

---

## Phase 1: test_cppfront_basic

**Objective**: Validate basic Cpp2 syntax handling (bounds-safety, span, while loops)

### Tasks

- [ ] Create git worktree `test/regression-01-basic`
- [ ] Run test_cppfront_basic in isolation
- [ ] Record results (stdout, stderr, exit code, timing)
- [ ] Verify test coverage against corpus files (pure2-bounds-safety-*.cpp2)
- [ ] Document any errors or gaps
- [ ] Merge worktree if fixes applied
- [ ] Cleanup worktree and branch
- [ ] Update validation report

**Expected Result**: PASS (currently passing)

**Corpus Files**: pure2-bounds-safety-span.cpp2

---

## Phase 2: test_cppfront_contracts

**Objective**: Validate contract/assertion handling

### Tasks

- [ ] Create git worktree `test/regression-02-contracts`
- [ ] Run test_cppfront_contracts in isolation
- [ ] Record results
- [ ] Verify coverage (pure2-contracts.cpp2)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: pure2-contracts.cpp2

---

## Phase 3: test_cppforward_functions

**Objective**: Validate function definitions, trailing commas, templates

### Tasks

- [ ] Create git worktree `test/regression-03-functions`
- [ ] Run test_cppforward_functions in isolation
- [ ] Record results
- [ ] Verify coverage (pure2-trailing-commas.cpp2)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: pure2-trailing-commas.cpp2

---

## Phase 4: test_cppforward_assertions

**Objective**: Validate assertion expressions and type casts

### Tasks

- [ ] Create git worktree `test/regression-04-assertions`
- [ ] Run test_cppforward_assertions in isolation
- [ ] Record results
- [ ] Verify coverage (pure2-assert-*.cpp2)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: pure2-assert-expected-not-null.cpp2

---

## Phase 5: test_cppforward_loops

**Objective**: Validate while/for loop constructs with next clauses

### Tasks

- [ ] Create git worktree `test/regression-05-loops`
- [ ] Run test_cppforward_loops in isolation
- [ ] Record results
- [ ] Verify coverage (loop patterns in corpus)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: Various pure2-*.cpp2 with loops

---

## Phase 6: test_cppforward_break_continue

**Objective**: Validate break/continue control flow

### Tasks

- [ ] Create git worktree `test/regression-06-break-continue`
- [ ] Run test_cppforward_break_continue in isolation
- [ ] Record results
- [ ] Verify coverage (pure2-break-continue.cpp2)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: pure2-break-continue.cpp2

---

## Phase 7: test_cppforward_fixed_type_aliases

**Objective**: Validate type alias definitions (fixed and template)

### Tasks

- [ ] Create git worktree `test/regression-07-type-aliases`
- [ ] Run test_cppforward_fixed_type_aliases in isolation
- [ ] Record results
- [ ] Verify coverage (mixed-fixed-type-aliases.cpp2)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: mixed-fixed-type-aliases.cpp2

---

## Phase 8: test_cppforward_function_expressions

**Objective**: Validate lambda/function expressions with std::for_each

### Tasks

- [ ] Create git worktree `test/regression-08-function-expressions`
- [ ] Run test_cppforward_function_expressions in isolation
- [ ] Record results
- [ ] Verify coverage (mixed-function-expression-*.cpp2)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: mixed-function-expression-and-std-for-each.cpp2

---

## Phase 9: test_cppforward_pointer_arithmetic

**Objective**: Validate pointer arithmetic and bounds safety

### Tasks

- [ ] Create git worktree `test/regression-09-pointers`
- [ ] Run test_cppforward_pointer_arithmetic in isolation
- [ ] Record results
- [ ] Verify coverage (bounds safety patterns)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: pure2-bounds-safety-*.cpp2

---

## Phase 10: test_cppforward_uninitialized_variables

**Objective**: Validate uninitialized variable detection

### Tasks

- [ ] Create git worktree `test/regression-10-uninit`
- [ ] Run test_cppforward_uninitialized_variables in isolation
- [ ] Record results
- [ ] Verify coverage (safety checker integration)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: Various (safety analysis patterns)

---

## Phase 11: test_cppforward_mixed_cpp1_cpp2

**Objective**: Validate mixed C++1 and Cpp2 syntax handling

### Tasks

- [ ] Create git worktree `test/regression-11-mixed`
- [ ] Run test_cppforward_mixed_cpp1_cpp2 in isolation
- [ ] Record results
- [ ] Verify coverage (mixed-allcpp1-hello.cpp2)
- [ ] Document errors/gaps (Note: parser may not support full mixed mode)
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: mixed-allcpp1-hello.cpp2

**Note**: This test passes but full corpus mixed files fail (50/189 blocked per regression_corpus_20251230)

---

## Phase 12: test_cppforward_string_interpolation

**Objective**: Validate string interpolation syntax $(var)

### Tasks

- [ ] Create git worktree `test/regression-12-string-interp`
- [ ] Run test_cppforward_string_interpolation in isolation
- [ ] Record results
- [ ] Verify coverage (string interpolation in corpus)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: Unknown (feature may not be in official cppfront)

---

## Phase 13: test_cppforward_inspect_pattern_matching

**Objective**: Validate inspect pattern matching (ranges, guards)

### Tasks

- [ ] Create git worktree `test/regression-13-pattern-match`
- [ ] Run test_cppforward_inspect_pattern_matching in isolation
- [ ] Record results
- [ ] Verify coverage (pure2-inspect-*.cpp2)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: pure2-inspect-*.cpp2

---

## Phase 14: test_cppforward_range_operators

**Objective**: Validate range operators (..<, ..=)

### Tasks

- [ ] Create git worktree `test/regression-14-ranges`
- [ ] Run test_cppforward_range_operators in isolation
- [ ] Record results
- [ ] Verify coverage (range-based loops in corpus)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: Unknown (may be extension)

---

## Phase 15: test_cppforward_performance_features

**Objective**: Validate move semantics and definite last use

### Tasks

- [ ] Create git worktree `test/regression-15-performance`
- [ ] Run test_cppforward_performance_features in isolation (10s timeout)
- [ ] Record results
- [ ] Verify coverage (move optimization patterns)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: Various (move semantics patterns)

---

## Phase 16: test_cppforward_error_handling

**Objective**: Validate try/catch error handling

### Tasks

- [ ] Create git worktree `test/regression-16-error-handling`
- [ ] Run test_cppforward_error_handling in isolation
- [ ] Record results
- [ ] Verify coverage (exception handling in corpus)
- [ ] Document errors/gaps
- [ ] Merge worktree if needed
- [ ] Cleanup
- [ ] Update report

**Expected Result**: PASS

**Corpus Files**: Various (exception patterns)

---

## Phase 17: Final Report and Cleanup

**Objective**: Generate comprehensive validation report

### Tasks

- [ ] Compile results from all 16 test validations
- [ ] Generate validation matrix (test x result x timing x coverage)
- [ ] Identify test coverage gaps vs. 189 corpus files
- [ ] Document corpus files not covered by any test
- [ ] Recommend new tests for uncovered patterns
- [ ] Archive validation artifacts
- [ ] Update conductor/tracks.md with completion status
- [ ] Close track

**Deliverables**:
- validation_report.md (comprehensive results)
- coverage_matrix.csv (test x corpus mapping)
- gaps_analysis.md (uncovered patterns)
- recommendations.md (suggested expansions)

---

## Notes

- All git worktrees created in parallel directory (../cppfort-test-*)
- Test isolation ensures no interference between validations
- Corpus preservation enforced (no .cpp2 file modifications)
- Timing recorded for regression detection
- Exit codes and stderr captured for debugging
