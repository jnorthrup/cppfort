# Fix Task Breakdown: Regression Recovery

**Generated:** 2025-10-30
**Scope:** 10 largest failing tests
**Expected Outcome:** 94.6% → <10% failure rate
**Effort Estimate:** 4-6 hours

---

## Phase 1: Pattern Recognition Enhancement (Tests 1, 3-10)

### Task 1.1: Add CPP2 Lambda Expression Pattern Handler
**File:** `/Users/jim/work/cppfort/src/stage0/unified_pattern_matcher.cpp`
**Complexity:** Medium
**Estimated Time:** 1.5 hours

**Objective:** Recognize and extract CPP2 lambda expressions

**Pattern to Recognize:**
```cpp2
:(args) -> return_type = { body }
```

**Extraction Requirements:**
1. Recognize opening: `:(` character sequence
2. Extract parameter list: `args` (may have implicit types)
3. Extract return type: `return_type` (or `_` for implicit)
4. Extract body: `{ body }` with balanced braces
5. Handle nesting: lambdas can be nested inside functions

**Implementation Checklist:**
```
[ ] Add `matches_cpp2_lambda()` method
[ ] Add `extract_cpp2_lambda_segments()` method
[ ] Handle implicit return types (when marked with `_`)
[ ] Handle parameter extraction (implicit and explicit types)
[ ] Test with single-level lambda
[ ] Test with nested lambdas
[ ] Test with multiple parameters
[ ] Test with inout/copy/move parameter modifiers
```

**Code Location Hints:**
- Look at `UnifiedPatternMatcher::find_matches()` (line ~34)
- Look at `try_match_at()` method (should exist)
- Add new pattern recognition in pattern matching loop

**Expected Generated Code After Fix:**
```cpp
auto get_next = [](auto iter) {
    // ... body content
};
```

---

### Task 1.2: Add CPP2 Function Declaration Pattern Handler
**File:** `/Users/jim/work/cppfort/src/stage0/unified_pattern_matcher.cpp`
**Complexity:** Medium
**Estimated Time:** 1.5 hours

**Objective:** Recognize and extract CPP2 function declarations with implicit parameter types

**Pattern to Recognize:**
```cpp2
name: (explicit_param: type, implicit_param) -> return_type = { body }
```

**Extraction Requirements:**
1. Recognize opening: function name followed by `:`
2. Extract function name
3. Extract parameter list with mixed explicit/implicit types
4. Extract return type
5. Extract function body
6. Preserve implicit parameter identities

**Implementation Checklist:**
```
[ ] Add `matches_cpp2_function()` method
[ ] Add `extract_cpp2_function_segments()` method
[ ] Handle parameter type inference from context
[ ] Handle return type inference (when marked with `_`)
[ ] Preserve parameter names for implicit types
[ ] Extract full function body with nesting
[ ] Test with single parameter
[ ] Test with multiple parameters (mixed explicit/implicit)
[ ] Test with generic/template parameters <T>
[ ] Test with requires clauses
```

**Code Location Hints:**
- Look at existing anchor-based extraction (lines ~140-150)
- Pattern will be: `$0: ($1) -> $2 = $3`
- But need semantic understanding of implicit types

**Expected Generated Code After Fix:**
```cpp
std::string create_result(std::string resultExpr, auto& r) {
    // ... function body
}
```

---

### Task 1.3: Implement Type Inference for Implicit Parameters
**File:** `/Users/jim/work/cppfort/src/stage0/unified_pattern_matcher.cpp`
**Complexity:** High
**Estimated Time:** 1.5 hours

**Objective:** Infer types for implicit parameter declarations

**Challenge:** CPP2 allows parameter types to be inferred from context

**Implementation Strategy:**

Option A: Simple heuristic approach
```cpp
std::string infer_param_type(const std::string& param_name,
                             const std::string& function_body,
                             const std::vector<std::pair<std::string,std::string>>& explicit_params) {
    // Look for usages of param_name in function_body
    // Match against known patterns:
    //   - r.group() → std::regex::smatch (or regex match type)
    //   - iter* → iterator (pointer to type)
    //   - to_char → char
    // Return best match or "auto" as fallback
}
```

Option B: Context-aware approach (requires full AST analysis)
```cpp
// This would require parsing the entire function context
// Too complex for current scope - defer to Phase 2
```

**Implementation Checklist:**
```
[ ] Add pattern usage analysis in function body
[ ] Create mapping of common patterns to types:
    [ ] Pattern: "r.group()" → Type: "std::regex::smatch&"
    [ ] Pattern: "iter*" → Type: iterator (from context)
    [ ] Pattern: method calls on param → infer from method usage
[ ] Add fallback to "auto" when inference fails
[ ] Document inferred types in generated comments
[ ] Test with regex test patterns
[ ] Test with last-use test patterns
```

**Code Location Hints:**
- Add helper function near extract methods
- Use regex pattern matching to identify usage
- Store inferred types in segment metadata

**Expected Generated Code After Fix:**
```cpp
// Regex test: parameter used as r.group(), r.group_start(), r.group_end()
std::string create_result(std::string resultExpr, auto& r) {
    // r is inferred as std::regex::smatch or similar
}
```

---

### Task 1.4: Update and Expand Test Coverage
**File:** `/Users/jim/work/cppfort/src/stage0/test_pattern_match.cpp`
**Complexity:** Low
**Estimated Time:** 1 hour

**Objective:** Add test cases for newly implemented patterns

**Test Cases to Add:**
```cpp
TEST(UnifiedPatternMatcher, cpp2_lambda_simple) {
    std::string pattern = ":(iter) -> _ = {";
    std::string input = "get_next := :(iter) -> _ = { ... }";
    auto result = UnifiedPatternMatcher::extract_segments(pattern, input);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->size(), 2);  // [params, body]
}

TEST(UnifiedPatternMatcher, cpp2_lambda_implicit_return) {
    std::string pattern = ":(inout iter) -> _ = {";
    std::string input = "extract_group := :(inout iter) -> _ = { ... }";
    auto result = UnifiedPatternMatcher::extract_segments(pattern, input);
    ASSERT_TRUE(result);
}

TEST(UnifiedPatternMatcher, cpp2_function_implicit_param) {
    std::string pattern = "create_result: (explicit: std::string, implicit) -> std::string = {";
    std::string input = "create_result: (resultExpr: std::string, r) -> std::string = { ... }";
    auto result = UnifiedPatternMatcher::extract_segments(pattern, input);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->size(), 4);  // [name, params, return, body]
}

TEST(UnifiedPatternMatcher, cpp2_nested_lambdas) {
    std::string pattern = "outer := :(x) -> _ = { inner := :(y) -> _ = { ... }";
    std::string input = actual_regex_test_pattern;
    auto result = UnifiedPatternMatcher::extract_segments(pattern, input);
    ASSERT_TRUE(result);
    // Verify nested lambda extracted
}

TEST(UnifiedPatternMatcher, cpp2_lambda_with_modifiers) {
    std::string pattern = "extract_until := :(inout iter, to: char) -> _ = {";
    std::string input = "extract_until := :(inout iter, to: char) -> _ = { ... }";
    auto result = UnifiedPatternMatcher::extract_segments(pattern, input);
    ASSERT_TRUE(result);
}

TEST(UnifiedPatternMatcher, type_inference_from_usage) {
    // Test that inferred type matches expected type
    std::string param_name = "r";
    std::string body = "r.group(0) and r.group_start(1)";
    std::string inferred = UnifiedPatternMatcher::infer_param_type(param_name, body, {});
    EXPECT_NE(inferred, "");
    EXPECT_NE(inferred, "auto");  // Should infer specific type
}
```

**Implementation Checklist:**
```
[ ] Add test for simple CPP2 lambda
[ ] Add test for lambda with multiple parameters
[ ] Add test for lambda with parameter modifiers (inout, copy, move)
[ ] Add test for CPP2 function with implicit parameter
[ ] Add test for nested lambdas
[ ] Add test for implicit return type inference
[ ] Add test for type inference from parameter usage
[ ] Run test suite to verify all pass
[ ] Test against actual regex test files
```

---

## Phase 2: Backchain Safety & Validation (Test 2)

### Task 2.1: Restore Conditional Backchain Mode
**File:** `/Users/jim/work/cppfort/src/stage0/rbcursive.cpp`
**Complexity:** Low
**Estimated Time:** 30 minutes

**Objective:** Only enable backchain when explicitly requested

**Current Problem:**
```cpp
// In 01489f2, backchain is unconditionally enabled
void speculate_backchain() {
    // Always called, always accumulates state
}
```

**Fix Implementation:**
```cpp
void speculate_backchain() {
    // Add environment variable check
    static bool backchain_enabled = [] {
        const char* env = std::getenv("RBCURSIVE_USE_BACKCHAIN");
        return env && std::string(env) == "1";
    }();

    if (!backchain_enabled) return;

    // ... existing backchain logic
}
```

**Implementation Checklist:**
```
[ ] Locate speculate_backchain() function
[ ] Add environment variable read
[ ] Cache the result (static variable)
[ ] Add early return if disabled
[ ] Verify normal tests pass with RBCURSIVE_USE_BACKCHAIN=0
[ ] Verify crash disappears
[ ] Keep backchain functionality for explicit enabling
```

---

### Task 2.2: Validate Segment Collection
**File:** `/Users/jim/work/cppfort/src/stage0/orbit_pipeline.cpp`
**Complexity:** Medium
**Estimated Time:** 1 hour

**Objective:** Prevent invalid segments from corrupting state

**Current Problem:**
```cpp
std::vector<std::string> collect_segments_from_traces(
    const std::vector<SemanticTrace>& traces
) {
    // No validation - returns whatever is collected
    // May contain uninitialized data
    // May have wrong count
    // May crash during processing
}
```

**Fix Implementation:**
```cpp
std::vector<std::string> collect_segments_from_traces(
    const std::vector<SemanticTrace>& traces
) {
    std::vector<std::string> segments;

    for (const auto& trace : traces) {
        // Validate trace before using it
        if (!trace.is_valid()) continue;
        if (trace.content.empty()) continue;

        // Collect segment
        segments.push_back(trace.content);
    }

    // Validate final result
    if (segments.empty()) {
        // Log warning or return empty
        std::cerr << "WARNING: collect_segments_from_traces returned empty\n";
    }

    return segments;
}
```

**Implementation Checklist:**
```
[ ] Identify trace validation method (is_valid() or equivalent)
[ ] Add validation check in loop
[ ] Handle invalid traces (skip or error)
[ ] Verify segment content before collecting
[ ] Add logging for debugging
[ ] Test with pure2-last-use.cpp2
[ ] Verify no crashes occur
[ ] Check that valid tests still pass
```

---

### Task 2.3: Add Initialization Guards
**File:** `/Users/jim/work/cppfort/src/stage0/confix_orbit.h` and `.cpp`
**Complexity:** Low
**Estimated Time:** 30 minutes

**Objective:** Prevent invalid segments from being used during orbit initialization

**Current Problem:**
```cpp
void ConfixOrbit::store_captured_segments(
    const std::vector<std::string>& segments
) {
    // No bounds checking
    // No content validation
    // May crash during later access
}
```

**Fix Implementation:**
```cpp
void ConfixOrbit::store_captured_segments(
    const std::vector<std::string>& segments
) {
    // Validate input
    if (segments.empty()) {
        std::cerr << "WARNING: Storing empty segments\n";
        // Don't store empty segments - keep previous valid state
        return;
    }

    // Validate each segment
    for (const auto& seg : segments) {
        if (seg.empty()) {
            std::cerr << "WARNING: Empty segment in captured segments\n";
            return;  // Reject entire batch if any segment invalid
        }
    }

    // Store validated segments
    this->m_captured_segments = segments;
}
```

**Implementation Checklist:**
```
[ ] Locate store_captured_segments() function
[ ] Add empty check
[ ] Add per-segment validation
[ ] Add error logging
[ ] Add early return on invalid data
[ ] Test with corrupted input
[ ] Verify graceful degradation (no crash)
[ ] Ensure valid segments still stored correctly
```

---

## Phase 3: Testing & Verification

### Task 3.1: Regression Test Suite Verification
**File:** Run regression tests
**Complexity:** Low
**Estimated Time:** 1 hour (includes execution time)

**Objective:** Verify fixes restore test pass rate

**Execution Steps:**
```bash
# Build with fixes
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run regression tests without backchain (Phase 1 fix)
cd regression-tests
RBCURSIVE_USE_BACKCHAIN=0 ./run_tests.sh
# Expected: Tests 1, 3-10 should pass

# Run regression tests with backchain (Phase 2 fix)
RBCURSIVE_USE_BACKCHAIN=1 ./run_tests.sh
# Expected: Test 2 should pass, Tests 1, 3-10 should pass
# Overall: >90% pass rate

# Run full test suite
make test
# Expected: All unit tests pass
```

**Verification Checklist:**
```
[ ] Pure2-regex tests (9 files) compile successfully
[ ] Pure2-last-use test transpiles without crash
[ ] Full regression suite runs to completion
[ ] Pass rate improves from 5.4% to >90%
[ ] No new test failures introduced
[ ] Build completes without errors
```

---

### Task 3.2: Add Integration Test Cases
**File:** Create new test file or add to existing
**Complexity:** Medium
**Estimated Time:** 1 hour

**Objective:** Ensure fixes are comprehensive and prevent regressions

**Test Cases:**
```cpp
// File: test_regex_integration.cpp
#include <gtest/gtest.h>
#include "unified_pattern_matcher.h"
#include "cpp2_emitter.h"

class RegexIntegrationTest : public ::testing::Test {
public:
    void SetUp() override {
        // Load regex test files
    }
};

TEST_F(RegexIntegrationTest, pure2_regex_14_compiles) {
    // Load pure2-regex_14_multiline_modifier.cpp2
    // Transpile to C++
    // Compile with C++ compiler
    EXPECT_EQ(compile_result, 0);
}

TEST_F(RegexIntegrationTest, pure2_regex_12_compiles) {
    // Similar for all 9 regex tests
}

TEST_F(RegexIntegrationTest, pure2_last_use_transpiles) {
    // Load pure2-last-use.cpp2
    // Transpile (should not crash)
    // Verify exit code is 0 (not -1)
    EXPECT_EQ(transpile_result, 0);
}

TEST_F(RegexIntegrationTest, backchain_disabled_safe) {
    // Run with RBCURSIVE_USE_BACKCHAIN=0
    // Verify no crashes
    EXPECT_EQ(test_result, 0);
}

TEST_F(RegexIntegrationTest, backchain_enabled_safe) {
    // Run with RBCURSIVE_USE_BACKCHAIN=1
    // Verify no crashes
    // Verify results same as without backchain
    EXPECT_EQ(test_result, 0);
}
```

**Implementation Checklist:**
```
[ ] Create test file for regex integration
[ ] Load each of 9 regex test files
[ ] Transpile each file
[ ] Compile resulting C++
[ ] Verify all 9 pass compilation
[ ] Load pure2-last-use.cpp2
[ ] Transpile (should not crash)
[ ] Test with RBCURSIVE_USE_BACKCHAIN=0
[ ] Test with RBCURSIVE_USE_BACKCHAIN=1
[ ] Verify no regressions in other tests
```

---

## Execution Order

### Priority 1: Pattern Recognition (Tests 1, 3-10) - Critical
```
Day 1, Hours 1-4:
  1. Task 1.1: Lambda expression handler (1.5 h)
  2. Task 1.2: Function declaration handler (1.5 h)
  3. Task 1.3: Type inference (1.5 h)
  4. Build and test incrementally (30 min)

Day 1, Hour 5:
  1. Task 1.4: Test coverage (1 h)
  2. Run test suite (30 min)
```

### Priority 2: Backchain Safety (Test 2) - High
```
Day 2, Hour 1:
  1. Task 2.1: Conditional backchain (30 min)
  2. Task 2.2: Segment validation (1 h)
  3. Task 2.3: Initialization guards (30 min)

Day 2, Hour 2:
  1. Build and test (30 min)
```

### Priority 3: Comprehensive Testing
```
Day 2, Hours 3-4:
  1. Task 3.1: Regression test suite (1 h including execution)
  2. Task 3.2: Integration tests (1 h)
  3. Final verification (30 min)
```

---

## Success Criteria

### Functional
```
✓ All 9 regex tests compile successfully
✓ pure2-last-use transpiles without crash (exit code 0)
✓ Full 185 test suite shows <10% failure rate (from 94.6%)
✓ No new test failures introduced
✓ Build completes without warnings
```

### Code Quality
```
✓ Pattern matching code is maintainable
✓ Type inference is well-documented
✓ Test coverage improved
✓ Error handling is robust
✓ No regressions from previous fixes
```

### Verification
```
✓ Run with RBCURSIVE_USE_BACKCHAIN=0 (normal mode)
✓ Run with RBCURSIVE_USE_BACKCHAIN=1 (backchain mode)
✓ Both modes produce same results
✓ No crashes in either mode
✓ Performance acceptable (no degradation)
```

---

**Task Breakdown Complete**
**Ready for Implementation**
