# Regression Test Failure Analysis

## Test Results: 1 PASS / 19 FAIL

### Root Cause Categories

#### 1. Multi-line Pattern Matching (6 failures)
**Tests:** pure2-type-safety-2, pure2-lifetime-safety-reject-null-error, pure2-assert-unique-ptr-not-null, pure2-type-safety-1, pure2-cpp1-prefix-expression-error, pure2-deducing-pointers-error

**Symptom:** Function signatures split across lines not transpiling
```cpp2
main: () -> int =
{
    // body
}
```

**Current behavior:** Line-by-line processing sees `main: () -> int =` as incomplete
**Transpiled (broken):** `main: () -> int =` left as-is

**Required fix:**
- Orbit-based processing with confix tracking
- Accumulate lines until balanced `{ }`
- Pattern match on complete construct

#### 2. Type Alias Syntax (4 failures)
**Tests:** pure2-union, pure2-is-with-polymorphic-types, pure2-bugfix-for-ufcs-noexcept, pure2-deduction-2-error

**Symptom:** `Name: type = ...` or `using  =` not transpiling
```cpp2
t: type = { /* members */ };
name_or_number: @union type = { /* variants */ };
```

**Current behavior:** Pattern `cpp2_type_alias` anchor `: type` not matching
**Transpiled (broken):** `tusing  = {;` (word boundary issue)

**Required fix:**
- Pattern for `identifier: type =`
- Handle metafunctions `@union`, `@polymorphic_base`
- Proper word boundary detection

#### 3. Missing Language Features (4 failures)
**Tests:** mixed-inspect-templates, mixed-is-as-value-with-variant, mixed-type-safety-1, pure2-regex_01_char_matcher

**Features not implemented:**
- `inspect expr -> Type { ... }` (pattern matching)
- `if v is (value)` (is operator)
- `forward` parameter type
- Template parameter syntax `<T : type>`
- Lambda expression `get_next :=(iter) { ... }`

**Required fix:** New patterns + semantic transforms

#### 4. Missing Type Definitions (2 failures)
**Tests:** pure2-bugfix-for-bad-capture-error, mixed-bugfix-for-cpp2-comment-cpp1-sequence

**Symptom:** `i32` type undefined
```cpp2
crash_10: (foo: i32) = { /* ... */ }
```

**Current behavior:** `i32` left as-is (cpp2 type alias)
**Required fix:** Type alias resolution or cpp2util.h header

#### 5. Missing Header Dependencies (2 failures)
**Tests:** pure2-types-smf-and-that-5, mixed-initialization-safety-1-error

**Symptom:** `#include "cpp2_inline.h"` not found
**Required fix:** Provide cpp2_inline.h or remove dependency

#### 6. Statement-Level Transforms (1 failure)
**Tests:** pure2-stdio-with-raii

**Symptom:** Assignment statements not in declaration position
```cpp2
myfile = fopen("xyzzy", "w");
_ = myfile.fprintf(...);
```

**Current behavior:** `myfile` treated as undeclared
**Required fix:** Statement-level pattern matching within function bodies

### Success Case
**Test:** simple_mixed
**Why it works:** Single-line functions with simple typed parameters

---

## Architecture Issues

### Current: Line-by-Line Processing
```
emit_depth_based() loops over lines
  → find_matches(line, patterns)
  → extract_segments(line, pattern, anchor_pos)
  → Fails if construct spans multiple lines
```

### Required: Orbit-Based Processing
```
1. Scan source for confix boundaries ({ }, ( ), [ ], < >)
2. Build confix_orbit tree
3. Extract evidence spans using orbit boundaries
4. Pattern match on complete orbits (not lines)
5. Apply transformations within orbit boundaries
```

### Evidence Span vs String Replacement

**Current (broken):**
```cpp
// String search and replace
result.find(" as ");
result.replace(pos, len, "std::get<...>");
```

**Required:**
```cpp
// Evidence span extraction
evidence_span before_as = extract_until(orbit, " as ");
evidence_span after_as = extract_after(orbit, " as ");
apply_transform(orbit, "as_operator", {before_as, after_as});
```

---

## Immediate Fixes by Impact

### High Impact (10+ tests)
1. **Multi-line function support** - Switch from line-by-line to orbit-based
2. **Type alias patterns** - Add `identifier: type = body`
3. **Word boundary validation** - Fix `using ` anchor matching

### Medium Impact (5-10 tests)
4. **inspect expressions** - New pattern + transform
5. **is operator** - New pattern + transform
6. **Template parameters** - `<T : type>` syntax

### Low Impact (<5 tests)
7. **forward parameter** - Existing pattern should work
8. **cpp2 type aliases** - Provide header or resolve
9. **Statement transforms** - Pattern matching in function bodies

---

## Orbit-Based Rewrite Requirements

### 1. Confix Orbit Scanner
```cpp
struct ConfixOrbit {
    char open, close;           // '{', '}' or '(', ')' etc
    size_t start_pos, end_pos;
    vector<ConfixOrbit*> children;
    string pattern_name;        // Matched pattern
};

ConfixOrbit* scan_orbits(string_view source);
```

### 2. Evidence Span Extractor
```cpp
struct EvidenceSpan {
    size_t start, end;
    string content;
    ConfixOrbit* parent_orbit;
};

vector<EvidenceSpan> extract_evidence(
    ConfixOrbit* orbit,
    const Pattern& pattern
);
```

### 3. Pattern Matcher with Orbits
```cpp
PatternMatch try_match_orbit(
    ConfixOrbit* orbit,
    const Pattern& pattern,
    const vector<EvidenceSpan>& evidence
);
```

### 4. Transform Application
```cpp
void apply_transform(
    ConfixOrbit* orbit,
    const string& transform_name,
    const vector<EvidenceSpan>& evidence
) {
    // Build output using evidence spans
    // Recursively transform child orbits
}
```

---

## Next Steps

1. Implement confix orbit scanner (reuse existing orbit_scanner.cpp?)
2. Rewrite pattern matching to work on orbits not lines
3. Add missing patterns (type alias, inspect, is)
4. Test on failures
