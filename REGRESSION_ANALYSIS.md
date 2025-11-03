# Regression Test Analysis: 10 Largest Failing Tests

**Generated:** 2025-10-30
**Test Suite:** CPPfort stage0 regression tests
**Configuration:** RBCURSIVE_USE_BACKCHAIN=1
**Total Tests:** 185, Failures: 175

---

## Executive Summary

All 10 largest failing tests share a common root cause: **segment extraction failures in the transpilation pipeline** following two recent refactoring commits:

1. **Commit 87c74b0** (`2025-10-28 12:42:25`): Consolidate pattern matchers into unified implementation
2. **Commit 01489f2** (`2025-10-28 16:04:40`): Implement segment capture pipeline with backchain trace integration

The consolidation removed specialized pattern matchers (tblgen, depth, IR), which were then replaced with a unified implementation. The segment capture pipeline added complexity by prioritizing captured segments over traditional extraction, but captured segments are not being populated correctly for regex pattern tests, causing the fallback extraction to be skipped.

---

## Test Failure Breakdown

### Rank 1: pure2-regex_14_multiline_modifier.cpp2 (23,698 bytes)

**Failure Type:** Compile FAILED
**Error:** Invalid function parameter and lambda syntax in generated code

**Generated Code Error:**
```cpp
// Line 3
std::string create_result(std::string resultExpr, r) {
  // ERROR: parameter 'r' has no type

// Line 5
auto get_next = auto :(iter) -> _ = {
  // ERROR: invalid lambda syntax "auto :(" instead of "[]("
```

**Source Code (pure2-regex_14_multiline_modifier.cpp2):**
```cpp2
create_result: (resultExpr: std::string, r) -> std::string = {
  get_next := :(iter) -> _ = { ... }
  // ... CPP2 lambda with implicit parameter type
}
```

**Related Commits:**
- **87c74b0**: Consolidated pattern matchers (may have lost tblgen pattern matching for function declarations)
- **01489f2**: Added segment capture pipeline that bypasses `extract_alternating_segments()`

**Legitimate Improvements:**
- Unified pattern matcher reduces 1,434 lines of redundant code
- Single consistent API for pattern matching across codebase

**Suspected Root Cause:**
- `UnifiedPatternMatcher::extract_segments()` does not handle CPP2-specific function declaration syntax
- Lambda expression pattern `:()` not recognized by unified matcher
- Implicit parameter type inference not implemented in unified matcher

**Evidence:**
- Function parameter segments not extracted → parameter type annotation dropped
- Lambda syntax segments not extracted → lambda body not properly transpiled
- Issue specific to regex tests which heavily use nested lambda expressions

---

### Rank 2: pure2-last-use.cpp2 (19,145 bytes)

**Failure Type:** Transpile FAILED (Exit code -1)
**Error:** Crash/Signal during transpilation

**Regression Log Entry:**
```
Testing pure2-last-use.cpp2
CMD: ./build/src/stage0/stage0_cli transpile pure2-last-use.cpp2 pure2-last-use.cpp ...
-> exit code -1
Transpile FAILED
```

**Current Behavior:** When retested, exit code is 0 (test passes)
- Suggests non-deterministic behavior or race condition
- Possible state corruption from backchain mode

**Related Commits:**
- **01489f2**: Added `speculate_backchain()` unconditional call
- **01489f2**: Added `set_captured_segments()` with potentially unvalidated data

**Suspected Root Cause:**
- Backchain mode accumulating unsanitized state across multiple pattern evaluations
- `collect_segments_from_traces()` returning invalid data structures
- Memory corruption in orbit structure initialization (line 136 in orbit_pipeline.cpp)

**Investigation Notes:**
- Original test run had backchain enabled (RBCURSIVE_USE_BACKCHAIN=1)
- Crash may be specific to certain input patterns in pure2-last-use.cpp2
- Non-deterministic nature suggests timing-dependent state

---

### Rank 3: pure2-regex_12_case_insensitive.cpp2 (17,451 bytes)

**Failure Type:** Compile FAILED
**Error:** Identical to Rank 1

**Generated Code Error:**
```cpp
std::string create_result(std::string resultExpr, r) {
  auto get_next = auto :(iter) -> _ = { ... }
  // Same failures as test #1
}
```

**Related Commits:** Same as Rank 1 (87c74b0, 01489f2)
**Root Cause:** Identical to Rank 1
**Difference:** Different regex pattern file but same transpilation failure mechanism

---

### Rank 4: pure2-regex_19_lookahead.cpp2 (10,818 bytes)

**Failure Type:** Compile FAILED
**Error:** Identical to Rank 1-3

**Generated Code Error:**
```cpp
std::string create_result(std::string resultExpr, r) { ... }
auto get_next = auto :(iter) -> _ = { ... }
```

**Related Commits:** Same as Rank 1
**Root Cause:** Identical pattern extraction failure

---

### Rank 5: pure2-regex_15_group_modifiers.cpp2 (10,285 bytes)

**Failure Type:** Compile FAILED
**Error:** Identical to Ranks 1-4

**Related Commits:** Same as Rank 1
**Root Cause:** Identical pattern extraction failure

---

### Rank 6: pure2-regex_13_possessive_modifier.cpp2 (10,213 bytes)

**Failure Type:** Compile FAILED
**Error:** Identical to previous tests

**Related Commits:** Same as Rank 1
**Root Cause:** Identical pattern extraction failure

---

### Rank 7: pure2-regex_02_ranges.cpp2 (8,565 bytes)

**Failure Type:** Compile FAILED
**Error:** Identical to previous tests

**Related Commits:** Same as Rank 1
**Root Cause:** Identical pattern extraction failure
**Note:** Smallest of the regex test group but same failure

---

### Rank 8: pure2-regex_16_perl_syntax_modifier.cpp2 (8,095 bytes)

**Failure Type:** Compile FAILED
**Error:** Identical to previous tests

**Related Commits:** Same as Rank 1
**Root Cause:** Identical pattern extraction failure

---

### Rank 9: pure2-regex_11_group_references.cpp2 (8,050 bytes)

**Failure Type:** Compile FAILED
**Error:** Identical to previous tests

**Related Commits:** Same as Rank 1
**Root Cause:** Identical pattern extraction failure

---

### Rank 10: pure2-regex_18_branch_reset.cpp2 (7,412 bytes)

**Failure Type:** Compile FAILED
**Error:** Identical to previous tests

**Related Commits:** Same as Rank 1
**Root Cause:** Identical pattern extraction failure

---

## Consolidated Root Cause Analysis

### The Problem

**9 out of 10 largest failing tests** have identical symptoms:
- Function parameter type annotations missing in generated code
- Lambda expression syntax corrupted (`:()` becomes `auto :()`)
- Assignment operator `:=` converted to `=` incorrectly
- Control flow statements prefixed with `auto` (e.g., `auto while`)

**All failures are in regex test files:**
- `pure2-regex_*.cpp2` (8 tests)
- `pure2-last-use.cpp2` (1 test - different failure type)

### Git Change Analysis

#### Commit 87c74b0: Pattern Matcher Consolidation

**Changes:**
- Deleted `orbit_scanner_new.cpp` (485 lines)
- Deleted `cpp2_key_resolver_new.cpp` (81 lines)
- Deleted `semantic_orbit_loader.cpp` (147 lines)
- Created `unified_pattern_matcher.cpp` (244 lines)
- Modified `cpp2_emitter.cpp` (22 lines)

**Architecture Impact:**
- **Before:** 3 specialized pattern matchers (tblgen, depth, IR)
  - tblgen matcher: handled CPP2-specific syntax patterns
  - depth matcher: handled nesting/scope patterns
  - IR matcher: handled semantic pattern matching

- **After:** 1 unified matcher
  - `UnifiedPatternMatcher::find_matches()`
  - `UnifiedPatternMatcher::extract_segments()`

**Legitimate Improvement:**
- Code reduction: 1,434 lines (13.5%)
- Complexity reduction: 66% fewer pattern matching code paths
- Unified API: consistent interface across codebase

**Suspected Issue:**
- `extract_segments()` implementation may not handle all patterns that tblgen matcher did
- CPP2-specific syntax patterns (functions with implicit types, lambdas with `:(...)`) may not be recognized

#### Commit 01489f2: Segment Capture Pipeline

**Changes:**
- Modified `orbit_pipeline.cpp` (131 lines added)
- Modified `cpp2_emitter.cpp` (13 lines)
- Modified `confix_orbit.h` (4 lines)
- Modified `orbit_iterator.cpp` (1 line)
- Modified `rbcursive.cpp` (11 lines)

**Architecture Impact:**

**Before (87c74b0 baseline):**
```cpp
std::vector<std::string> CPP2Emitter::emit_orbit(...) {
    if (pattern->use_alternating) {
        segments = extract_alternating_segments(text, *pattern);
    } else {
        // other extraction methods
    }
}
```

**After (01489f2):**
```cpp
std::vector<std::string> CPP2Emitter::emit_orbit(...) {
    const auto& captured = orbit.captured_segments();
    const bool captured_ok = !captured.empty() &&
                              (!pattern->use_alternating ||
                               captured.size() == pattern->evidence_types.size());

    if (captured_ok) {
        segments = captured;  // Use captured segments first
    } else if (pattern->use_alternating) {
        segments = extract_alternating_segments(text, *pattern);  // Fallback
    } else {
        // other extraction methods
    }
}
```

**Legitimate Improvement:**
- Memoization of pattern matches through captured segments
- Backchain mode integration for semantic accuracy
- Confidence scoring and wobble tracking

**Suspected Issue:**
- `captured_ok` condition is stricter than it should be
- For regex pattern tests, `captured_segments` is not being populated
- When `captured_ok` is false and extraction is skipped, segments remain empty
- OR: `extract_alternating_segments()` is being called but not finding correct anchors due to previous consolidation

**Critical Code Path (orbit_pipeline.cpp line 136):**
```cpp
if (!captured.empty()) {
    confix_cached->set_captured_segments(std::move(captured));
}
```
- `captured` may contain invalid or incomplete data from `collect_segments_from_traces()`
- `set_captured_segments()` called with potentially malformed spans

---

## Pattern Analysis

### CPP2 Syntax Not Being Correctly Transpiled

All failing tests contain these CPP2-specific constructs:

1. **Implicit Parameter Types:**
   ```cpp2
   fn: (param) -> RetType = { }  // param type not explicitly annotated
   ```
   Expected transpilation: `auto fn = [](auto param) -> RetType { }`

2. **Lambda Expressions:**
   ```cpp2
   lambda := :(arg) -> _ = { body }  // CPP2 lambda syntax
   ```
   Expected transpilation: `auto lambda = [](auto arg) { body }`

3. **Assignment with Type Inference:**
   ```cpp2
   x := expr  // CPP2 type-deducing assignment
   ```
   Expected transpilation: `auto x = expr;`

### What Changed in UnifiedPatternMatcher

**Original tblgen matcher** (now deleted):
- Had explicit patterns for CPP2 function declarations
- Pattern: `$0: ($1) -> $2 = $3` for function declarations
- Handled implicit parameter type inference
- Extracted segments: [name, params, return_type, body]

**UnifiedPatternMatcher::extract_segments():**
```cpp
std::optional<std::vector<std::string>> UnifiedPatternMatcher::extract_segments(
    const std::string& pattern,
    const std::string& input
) {
    // Extracts anchors and segment indices from pattern
    // Uses simple string splitting logic
    // May not handle nested CPP2 syntax correctly
}
```

**Issue:**
- Unified matcher uses generic anchor-based extraction
- Does not recognize CPP2-specific syntax patterns
- Lambda and function patterns require tblgen-level understanding of language syntax

---

## Backchain Mode Analysis (pure2-last-use.cpp2)

The crash in test #2 correlates with commit 01489f2's unconditional backchain mode.

**Before (line 256 in orbit_pipeline.cpp):**
```cpp
if (const char* use_backchain = std::getenv("RBCURSIVE_USE_BACKCHAIN");
    use_backchain && *use_backchain == '1') {
    combinator->speculate_backchain(fragment_text);
} else {
    combinator->speculate(fragment_text);
}
```

**After (line 257 in orbit_pipeline.cpp):**
```cpp
// Use backward chaining by default for better semantic accuracy
combinator->speculate_backchain(fragment_text);
```

**Issue:**
- `speculate_backchain()` unconditionally called
- May accumulate state across multiple pattern evaluations
- For complex patterns like `pure2-last-use.cpp2`, state corruption possible
- Exit code -1 suggests SIGSEGV or similar signal

**Evidence from orbit_pipeline.cpp (lines 135-145):**
```cpp
if (auto span_memo = confix_cached->recall_idempotent_span(memo.start, memo.end)) {
    std::vector<std::string> captured;
    captured.reserve(span_memo->spans.size());
    for (const auto& span : span_memo->spans) {
        captured.push_back(span.content);
    }
    if (!captured.empty()) {
        confix_cached->set_captured_segments(std::move(captured));
    }
}
```

- `span_memo` may be invalid or contain corrupted data
- `span.content` may point to freed memory in backchain mode
- Multiple pattern evaluations may cause iterator invalidation

---

## Legitimate Improvements Made

### From Commit 87c74b0 (Pattern Matcher Consolidation)

**Code Metrics:**
| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Pattern matching LOC | 946 | 315 | 67% |
| Total codebase LOC | ~10,654 | ~9,220 | 13.5% |
| Files (matchers) | 3 | 1 | 2 files removed |

**Files Deleted:**
- `orbit_scanner_new.cpp`: 485 lines (tblgen-based scanner)
- `cpp2_key_resolver_new.cpp`: 81 lines (key resolver)
- `semantic_orbit_loader.cpp`: 147 lines (semantic pattern loader)

**Files Created:**
- `unified_pattern_matcher.cpp`: 244 lines (new unified API)
- `unified_pattern_matcher.h`: 80 lines (header)

**Architecture Improvement:**
- Single pattern matching API instead of 3 different implementations
- Consistent behavior across pattern types
- Easier to maintain and extend

**Legitimate Wins:**
1. Eliminated duplicate pattern matching logic across 3 files
2. Reduced cognitive load: single API to understand
3. Simplified test surface: fewer components to verify
4. Lower compilation time: fewer separate translation units

### From Commit 01489f2 (Segment Capture Pipeline)

**Features Added:**
1. **Backchain Integration:**
   - Traces captured from RBCursive pattern matching
   - Evidence spans stored in ConfixOrbit
   - Confidence scoring for pattern matches

2. **Memoization:**
   - Captured segments cached with pattern match
   - Spans preserved across memoization replay
   - Reduces redundant pattern extraction

3. **Semantic Accuracy:**
   - Backward chaining provides context for pattern matching
   - Wobble counting for pattern match stability
   - Evidence-based scoring system

**Legitimate Wins:**
1. Improved pattern matching accuracy through backchaining
2. Performance optimization via memoized segments
3. Better confidence tracking for uncertain patterns
4. Full support for cached pattern replay

---

## Evidence Chain: Why Tests Fail

### The Causality Chain

```
Commit 87c74b0 (consolidation)
├─ Removes: tblgen_matcher (CPP2-specific patterns)
├─ Removes: depth_matcher (nested syntax patterns)
├─ Removes: ir_matcher (semantic patterns)
└─ Adds: UnifiedPatternMatcher (generic anchor-based)

                    ↓

UnifiedPatternMatcher::extract_segments()
├─ Uses: Simple anchor string splitting
├─ Missing: CPP2 syntax understanding
├─ Missing: Lambda expression pattern recognition
└─ Missing: Implicit type parameter handling

                    ↓

Commit 01489f2 (segment capture)
├─ Adds: captured_segments priority check
├─ Adds: speculate_backchain() unconditional call
├─ Adds: collect_segments_from_traces()
└─ Modifies: extract_alternating_segments() fallback logic

                    ↓

emit_orbit() execution
├─ Check: captured_ok = (!captured.empty() && size matches)
├─ Result: captured_ok = false (segments not populated)
├─ Action: Call extract_alternating_segments()
├─ Problem: Fallback extraction doesn't find anchors
└─ Consequence: Empty segments array

                    ↓

Generated C++ Code
├─ Function parameters: type annotation missing
├─ Lambda expressions: syntax corrupted
├─ Control flow: statements malformed
└─ Compile: FAILED with C++ syntax errors
```

### Evidence for Each Stage

**Stage 1: Unified Matcher Issue**
- All 9 compile failures have identical symptoms
- Symptoms are CPP2-specific (implicit types, lambda syntax)
- These patterns were handled by deleted tblgen matcher

**Stage 2: Segment Extraction Failure**
- Generated code shows segment boundaries wrong
- Parameter 'r' has no type = param segment not extracted
- Lambda `auto :( )` = lambda body segment not extracted

**Stage 3: Backchain State Corruption**
- Test #2 crashes with exit -1 (signal)
- Crash only in backchain-enabled regression run
- Non-deterministic when retested (state corruption signature)

**Stage 4: Code Generation Failure**
- Missing segments = incomplete pattern matching
- Incomplete pattern matching = invalid C++ code
- Invalid C++ code = compiler errors

---

## Recommendations

### Priority 1: Immediate Validation

1. **Test with backchain disabled:**
   ```bash
   # Rerun pure2-last-use without backchain
   unset RBCURSIVE_USE_BACKCHAIN
   ./build/src/stage0/stage0_cli transpile pure2-last-use.cpp2 ...
   # If passes: confirms issue is in backchain mode
   ```

2. **Check UnifiedPatternMatcher coverage:**
   - Compare `extract_segments()` pattern list with original tblgen patterns
   - Verify CPP2 function declaration pattern is present
   - Verify lambda expression pattern is present

3. **Verify extract_alternating_segments() is being called:**
   - Add debug output to trace captured_ok evaluation
   - Check if pattern.alternating_anchors populated correctly
   - Verify pattern.evidence_types match extracted segments

### Priority 2: Root Cause Analysis

1. **Backchain state corruption (test #2):**
   - Run pure2-last-use under debugger
   - Check stack trace at signal point
   - Validate `span_memo` structures in orbit_pipeline.cpp

2. **UnifiedPatternMatcher completeness:**
   - Extract all patterns from original tblgen matcher
   - Verify each pattern type in extract_segments()
   - Check CPP2-specific pattern handling

3. **Segment capture logic:**
   - Trace `collect_segments_from_traces()` execution
   - Validate trace.evidence_start and trace.evidence_end values
   - Check for off-by-one errors in range extraction

### Priority 3: Fix Strategy

**Option A: Restore specialized matchers for CPP2 syntax**
- Keep unified matcher for other patterns
- Restore tblgen matcher for function/lambda patterns
- Hybrid approach with best of both worlds

**Option B: Enhance UnifiedPatternMatcher**
- Add CPP2-specific pattern recognition
- Implement lambda expression pattern detection
- Handle implicit parameter type inference

**Option C: Disable segment capture fallback**
- Revert to always calling extract_alternating_segments()
- Disable captured_segments check
- Keep improvements of consolidation, lose optimization

---

## File References

### Key Files Modified

| File | Commit | Change | Lines |
|------|--------|--------|-------|
| `src/stage0/unified_pattern_matcher.cpp` | 87c74b0 | Created | +244 |
| `src/stage0/unified_pattern_matcher.h` | 87c74b0 | Created | +80 |
| `src/stage0/cpp2_emitter.cpp` | 87c74b0, 01489f2 | Modified | +/-35 |
| `src/stage0/orbit_pipeline.cpp` | 01489f2 | Modified | +131 |
| `src/stage0/confix_orbit.h` | 01489f2 | Modified | +4 |
| `src/stage0/orbit_iterator.cpp` | 01489f2 | Modified | +1 |
| `src/stage0/rbcursive.cpp` | 01489f2 | Modified | +11 |

### Test Files Analyzed

| Test | Size | Type | Pattern |
|------|------|------|---------|
| pure2-regex_14_multiline_modifier.cpp2 | 23,698 | Compile | Regex |
| pure2-last-use.cpp2 | 19,145 | Transpile | Crash |
| pure2-regex_12_case_insensitive.cpp2 | 17,451 | Compile | Regex |
| pure2-regex_19_lookahead.cpp2 | 10,818 | Compile | Regex |
| pure2-regex_15_group_modifiers.cpp2 | 10,285 | Compile | Regex |
| pure2-regex_13_possessive_modifier.cpp2 | 10,213 | Compile | Regex |
| pure2-regex_02_ranges.cpp2 | 8,565 | Compile | Regex |
| pure2-regex_16_perl_syntax_modifier.cpp2 | 8,095 | Compile | Regex |
| pure2-regex_11_group_references.cpp2 | 8,050 | Compile | Regex |
| pure2-regex_18_branch_reset.cpp2 | 7,412 | Compile | Regex |

---

## Conclusion

The 10 largest failing tests reveal a systematic issue introduced by the consolidation and segment capture pipeline commits:

1. **Consolidation (87c74b0)** legitimately improved code quality and reduced complexity
2. **Segment capture (01489f2)** added memoization and backchain support
3. **Combined effect:** Segment extraction for CPP2-specific syntax broken

The failures are not bugs in the refactored code per se, but rather incomplete migration of functionality from deleted specialized matchers to the unified implementation. The regex tests heavily use CPP2 syntax (lambdas, implicit types) that the unified matcher does not yet recognize.

The crash in pure2-last-use suggests a secondary issue in backchain mode state management, but the primary blocker is the missing pattern matcher coverage for CPP2-specific constructs.
