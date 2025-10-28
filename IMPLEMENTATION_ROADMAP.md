# Implementation Roadmap - No Cheating Edition

## Core Principle: One Feature at a Time, Fully Working

### Current Honest Status
- **Working Features**: 0
- **Passing Tests**: 0/192
- **Lines of Working Code**: ~0 (infrastructure exists but doesn't produce correct output)
- **Can Transpile Anything Correctly**: NO

## Stage 1: Make ANYTHING Work (Week 1)

### Goal: Pass ONE test completely

#### Step 1.1: Simplest Possible Test
```cpp2
// test_minimal.cpp2
main: () -> int = {
    return 0;
}
```

Expected output:
```cpp
int main() {
    return 0;
}
```

#### Implementation Tasks:
1. [ ] Parse function signature (name, params, return type)
2. [ ] Extract function body
3. [ ] Preserve return statement
4. [ ] Generate correct C++ syntax
5. [ ] Verify byte-for-byte match with expected output

#### Success Criteria:
- Test passes 100% of the time
- No hardcoding
- No regex post-processing hacks
- Pattern-driven transformation

### STOP HERE until Step 1.1 is 100% complete

## Stage 2: Variable Declarations (Week 2)

### Goal: Transform variable declarations

#### Step 2.1: Typed Variables
```cpp2
x: int = 42;  →  int x = 42;
s: std::string = "hello";  →  std::string s = "hello";
```

#### Step 2.2: Auto Variables (Walrus)
```cpp2
x := 42;  →  auto x = 42;
s := "hello";  →  auto s = "hello";
```

#### Implementation Tasks:
1. [ ] Pattern for `: type =` syntax
2. [ ] Pattern for `:=` syntax
3. [ ] Segment extraction for name, type, initializer
4. [ ] Substitution to C++ syntax
5. [ ] Test both patterns in function bodies

### STOP HERE until both patterns work

## Stage 3: Parameters (Week 3)

### Goal: Transform function parameters

#### Step 3.1: Basic Parameters
```cpp2
foo: (x: int, y: double) -> void
```
→
```cpp
void foo(int x, double y)
```

#### Step 3.2: Parameter Passing Modes
```cpp2
inout → &
in → const
out → &
move → &&
forward → &&
```

#### Implementation Tasks:
1. [ ] Parse parameter list
2. [ ] Extract parameter mode, name, type
3. [ ] Transform to C++ parameter syntax
4. [ ] Handle multiple parameters
5. [ ] Test all parameter modes

## Stage 4: Include Generation (Week 4)

### Goal: Generate and preserve includes

#### Tasks:
1. [ ] Detect std:: usage
2. [ ] Map types to headers
3. [ ] Generate #include directives
4. [ ] Place at top of file
5. [ ] Avoid duplicates

## Regression Test Tracking

### Test Categories:
1. **Trivial** (5 tests) - Empty functions, single statements
2. **Simple** (20 tests) - Variables, basic functions
3. **Moderate** (50 tests) - Parameters, multiple functions
4. **Complex** (50 tests) - Templates, contracts
5. **Advanced** (67 tests) - Full CPP2 features

### Progression Requirements:
- Must pass 80% of previous category before moving to next
- Document why each test fails
- No skipping to easier tests in harder categories

## Anti-Cheat Mechanisms

### 1. Output Validation
Every transformation must:
- [ ] Compile with g++ -std=c++20
- [ ] Produce identical behavior to reference
- [ ] Pass without any post-processing
- [ ] Work for variations of the pattern

### 2. Progress Gates
Cannot proceed to next stage until:
- [ ] Current stage tests pass
- [ ] Code review confirms no hacks
- [ ] Variations of patterns tested
- [ ] No hardcoded solutions

### 3. Honesty Metrics
Track and report:
- Real passing test count (not "should pass")
- Actual transformation count (not "infrastructure exists")
- Lines of code that produce correct output
- Features that work end-to-end

## What NOT to Do

### ❌ Architecture Astronautics
- No more "semantic codec" design
- No more "n-way graph mapping" theory
- No more "orbit recursion" without implementation
- No more phases beyond Stage 4

### ❌ Fake Progress
- Don't mark items complete without tests passing
- Don't claim "working" when it crashes
- Don't use regex hacks and call it "working"
- Don't report 100% confidence when nothing works

### ❌ Scope Creep
- Don't add bidirectional patterns yet
- Don't design for self-hosting yet
- Don't optimize for performance yet
- Don't refactor working code yet

## Success Metric: The Truth Test

Can you run this and get correct output?
```bash
./stage0_cpp2 test_minimal.cpp2 > output.cpp
g++ -std=c++20 output.cpp -o test
./test
echo $?  # Should be 0
```

If NO → Keep working on Stage 1
If YES → Move to next test

## Daily Progress Report Format

```
Date: YYYY-MM-DD
Tests Passing: X/192
Features Complete: [List actual working features]
Current Stage: X.X
Blocking Issue: [What's preventing the next test from passing]
Lines Changed: +X -Y
Honest Assessment: [Can it transpile anything useful?]
```

## The Only Goal That Matters

**Make it work, make it right, make it fast - IN THAT ORDER**

Currently we're at step 0: Nothing works.
Let's get to step 1: Something works.
Everything else can wait.