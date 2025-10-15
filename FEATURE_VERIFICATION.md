# Feature Verification Framework

## Problem Statement
TODO.md claims various features work (marked with ✓) but regression tests show 0/192 passing. This indicates a massive gap between claimed functionality and reality.

## Required Features for Basic Transpilation

### Level 0: Minimal Viable Feature Set
These are REQUIRED to pass even the simplest test:

1. **Function Declaration Parsing**
   - Input: `main: () -> int = { }`
   - Required: Parse name, parameters, return type, body
   - Status: UNKNOWN (claimed working but untested)

2. **Function Body Processing**
   - Input: `{ s: std::string = "hello"; }`
   - Required: Transform variable declarations inside body
   - Status: PARTIALLY WORKING (per TODO.md example)

3. **Include Preservation**
   - Input: File with `#include <iostream>`
   - Required: Keep includes at top of output
   - Status: NOT IMPLEMENTED

### Level 1: Basic Parameter Support
Required for `mixed-hello.cpp2`:

1. **Inout Parameter**
   - Input: `inout s: std::string`
   - Output: `std::string& s`
   - Status: NOT IMPLEMENTED (explicitly listed as gap)

2. **Function Ordering**
   - Input: `decorate` called before definition
   - Output: Forward declaration or reordering
   - Status: NOT IMPLEMENTED

### Level 2: Type System
Required for any non-trivial code:

1. **Auto Type Deduction**
   - Input: `x := 42`
   - Output: `auto x = 42`
   - Status: UNKNOWN

2. **Template Syntax**
   - Input: `type Pair<A,B> = std::pair<A,B>`
   - Output: `template<typename A, typename B> using Pair = std::pair<A,B>`
   - Status: BROKEN (malformed output per TODO.md)

## Verification Tests

### Test 0: Absolute Minimum
```cpp2
// Input
main: () -> int = { }

// Expected Output
int main() { }
```
**Verification**: Can we transform an empty main function?

### Test 1: Single Statement
```cpp2
// Input
main: () -> int = {
    return 0;
}

// Expected Output
int main() {
    return 0;
}
```
**Verification**: Can we preserve a return statement?

### Test 2: Variable Declaration
```cpp2
// Input
main: () -> int = {
    x: int = 42;
}

// Expected Output
int main() {
    int x = 42;
}
```
**Verification**: Can we transform a typed variable declaration?

### Test 3: Walrus Operator
```cpp2
// Input
main: () -> int = {
    x := 42;
}

// Expected Output
int main() {
    auto x = 42;
}
```
**Verification**: Can we transform auto type deduction?

## Implementation Gaps Analysis

### What's Actually Implemented:
Based on code review and TODO.md:

1. **Pattern Loading**: YAML patterns load but don't work correctly
2. **Orbit Infrastructure**: Classes exist but aren't properly connected
3. **One Transformation**: `main: () -> int = {}` might partially work
4. **Substitution Bug**: Template alias produces garbage output

### What's Completely Missing:
1. Parameter transformations (in/out/inout/move/forward)
2. Include generation/preservation
3. Forward declarations
4. Bidirectional patterns
5. Recursive pattern application (using regex hack)
6. Contract syntax
7. Inspect/is/as expressions

### What's Broken:
1. All 192 regression tests fail
2. Template type alias substitution
3. Confidence scoring (hardcoded/meaningless)
4. Grammar classification (may not work correctly)

## Action Items

### Priority 1: Make ONE Test Pass
Before adding ANY new features:

1. Pick `test_simple.cpp2` (if it exists) or create minimal test
2. Run current transpiler on it
3. Compare actual vs expected output
4. Fix ONLY what's needed for that test
5. Document the fix
6. Verify test passes consistently

### Priority 2: Fix Parameter Transformation
This blocks many tests:

1. Implement parameter parser
2. Handle `inout` → `&` transformation
3. Handle `move` → `&&` transformation
4. Test on `mixed-hello.cpp2`

### Priority 3: Include Handling
Required for standard library usage:

1. Preserve existing includes
2. Detect std:: usage
3. Generate missing includes
4. Place at top of file

## Metrics for Success

### Current State (Honest):
- Passing tests: 0/192 (0%)
- Features working: ~10%
- Can transpile hello world: NO
- Can self-host: NO

### Milestone 1 (Minimal):
- Passing tests: 1/192 (0.5%)
- Features working: 20%
- Can transpile hello world: YES
- Can self-host: NO

### Milestone 2 (Basic):
- Passing tests: 10/192 (5%)
- Features working: 40%
- Can transpile simple programs: YES
- Can self-host: NO

### Milestone 3 (Functional):
- Passing tests: 50/192 (26%)
- Features working: 60%
- Can transpile most programs: YES
- Can self-host: PARTIAL

### Milestone 4 (Complete):
- Passing tests: 150+/192 (78%+)
- Features working: 90%+
- Can transpile complex programs: YES
- Can self-host: YES

## No More Architecture Astronautics

STOP:
- Adding new phases to TODO.md
- Creating new abstract classes
- Designing "semantic codecs"
- Writing about "n-way graph mapping"

START:
- Making one test pass
- Fixing actual bugs
- Implementing missing features
- Measuring real output

The path forward is not more architecture - it's making the existing code actually work.