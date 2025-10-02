# Sprint Status - Orbit Scanner Truthfulness & Emitter Fixes

**Date:** 2025-10-01
**Sprint Goal:** Truth-based orbit detection + working cpp2→C++ transpiler
**Status:** 🟡 PARTIAL - Orbit scanner complete, emitter blocked

---

## ✅ Completed This Sprint

### Orbit Scanner Truth-Based Detection

- **Depth-aware pattern matching** implemented & tested
  - Validates patterns only match at correct nesting levels
  - Confix-aware: patterns respect `{}`, `()`, `<>`, `[]`, `""` context
  - String literal masking: no false matches inside quotes
  - 4/4 unit tests passing ([tests/test_orbit_scanner_with_mocks.cpp:349-450](tests/test_orbit_scanner_with_mocks.cpp:349))

- **Confidence scoring truthfulness**
  - Signature matches = 0.95 confidence (actual truth)
  - Heuristics are fallback only (documented at [src/stage0/orbit_scanner.cpp:180-182](src/stage0/orbit_scanner.cpp:180))
  - No fabrication: removed forced threshold boosting

- **Pattern infrastructure**
  - Added `expected_depth` and `required_confix` to OrbitPattern
  - cpp2 signature patterns: `operator=:`, `out`, `inout`, `: i32`, etc.
  - Depth validation prevents matches at wrong scope levels

### Type System Improvements

- **Wildcard type `_` conversion** to `auto`
  - Parameter types: 40 errors → 4 errors
  - Return types: handled
  - [src/stage0/emitter.cpp:779,806,836](src/stage0/emitter.cpp:779)

---

## 🔴 Blocked / Critical Issues

### Emitter Expression Corruption

**Severity:** HIGH - Blocks 91.5% of regression tests

**Bug:** [src/stage0/emitter.cpp:601-602](src/stage0/emitter.cpp:601)

```cpp
::std::regex suffix_deref(R"((\w+)\*(?=\s*(?:[+\-;,\)\]\}]|$)))");
expr = ::std::regex_replace(expr, suffix_deref, "*$1");
```

**Impact:** Corrupts qualified names

- Input: `std::string{"Hello, "} + std::string{name}`
- Output: `std:::std::string{"Hello, "} +::string{name}` ❌
- Test case: [examples/stage0/hello.cpp2:9](examples/stage0/hello.cpp2:9) → [hello_output.cpp:34](hello_output.cpp:34)

**Root Cause:** Regex matches `\w+` across `::` boundaries, breaking namespace resolution

---

## 📊 Metrics

### Regression Tests

- **Current:** 11/130 pass (8.5%)
- **Target:** >15% pass rate this sprint
- **Blockers:** 77 parse failures, 42 compile failures (mostly expression mangling)

### Test Coverage

- Orbit scanner: 4/4 depth filtering tests ✅
- End-to-end: Simple cases work, complex expressions broken ❌

---

## 🎯 Next Sprint Priorities

### Must Fix (P0)

1. **Expression regex safety** - Fix suffix_deref to preserve qualified names
2. **Regression validation** - hello.cpp2 must compile cleanly
3. **Pass rate improvement** - Achieve >15% regression success

### Should Fix (P1)

4. Class member syntax (`name: type` inside classes)
5. Operator syntax (`operator=:` → constructors)
6. String interpolation (`(var)$` patterns)

### Nice to Have (P2)

7. Confix visibility bitmask (full masking matrix)
8. Parser improvements for complex syntax
9. Template/constraint handling

---

## 🚧 Risks & Dependencies

### Technical Debt

- Emitter has accumulated fragile regex transformations
- Parser rejects many valid cpp2 constructs (77 failures)
- No integration tests for end-to-end pipeline

### Blockers

- **Cannot claim working transpiler until emitter fixed**
- Expression corruption prevents real-world cpp2 usage
- Orbit improvements invisible until emitter stable

---

## 📝 Notes for Next Developer

### What Works Right Now

- Simple cpp2 files with basic syntax compile
- Orbit scanner correctly validates depth/confix
- Type aliases and parameter modes convert properly
- Mock infrastructure enables fast iteration

### What's Broken

- **Do not trust** hello_output.cpp line 34 - it's invalid C++
- Regex transformations conflict with each other
- Claims of "working pipeline" were premature - only 8.5% pass rate

### Quick Win Path

1. Fix one regex at a time in [emitter.cpp](src/stage0/emitter.cpp)
2. Test against [hello.cpp2](examples/stage0/hello.cpp2) after each fix
3. Run regression suite to measure real progress
4. Don't claim victory until actual cpp2 code compiles

### Files to Review

- [src/stage0/emitter.cpp:590-640](src/stage0/emitter.cpp:590) - Expression transformation logic
- [src/stage0/orbit_scanner.cpp:132-166](src/stage0/orbit_scanner.cpp:132) - Depth validation
- [tests/test_orbit_scanner_with_mocks.cpp:347-451](tests/test_orbit_scanner_with_mocks.cpp:347) - Truth validation tests

---

**Summary:** Orbit scanner is truth-based and tested. Emitter corrupts expressions. Need regex fix before claiming working transpiler.
