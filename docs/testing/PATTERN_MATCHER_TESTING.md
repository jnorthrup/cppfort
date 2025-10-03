# Pattern Matcher Testing Guide

## Overview

This document describes the comprehensive test harness for the n-way lowering pattern infrastructure in cppfort. The test suite validates pattern matching, multi-target code generation, and machine abstraction layers.

## Test Structure

### Test Files

1. **test_pattern_matcher.cpp** - Core pattern matcher unit tests
2. **test_pattern_integration.cpp** - Integration tests for pattern pipelines
3. **test_machine_patterns.cpp** - Machine abstraction and dialect-specific tests
4. **test_pattern_lowering.sh** - Regression test suite

## Test Coverage

### 1. Pattern Matcher Unit Tests

**File:** `tests/test_pattern_matcher.cpp`

#### Basic Pattern Registration
- Pattern registration for single target
- Multiple patterns for same NodeKind
- Patterns across multiple target languages
- Pattern existence checking

#### Priority System
- Pattern priority ordering
- Multiple patterns with different priorities
- Priority-based pattern selection

#### Pattern Retrieval
- Getting patterns by NodeKind
- Pattern count tracking
- Empty pattern queries

#### Multi-Target Lowering
- Arithmetic operations to MLIR Arith dialect
- Bitwise operations to MLIR Arith dialect
- Comparison operations to MLIR Arith dialect
- Control flow to MLIR CF dialect
- Structured control flow to MLIR SCF dialect
- Memory operations to MLIR MemRef dialect
- Function operations to MLIR Func dialect

#### Pattern Constraints
- Type constraint validation
- CFG constraint validation
- Combined constraint testing

#### Coverage Tests
- All arithmetic operations (ADD, SUB, MUL, DIV, MOD, NEG, ABS)
- All bitwise operations (AND, OR, XOR, SHL, ASHR, LSHR, NOT)
- All comparison operations (EQ, NE, LT, LE, GT, GE)
- All floating point operations (FADD, FSUB, FMUL, FDIV, FNEG, FABS)

### 2. Pattern Integration Tests

**File:** `tests/test_pattern_integration.cpp`

#### Full Pipeline Tests
- Complete arithmetic lowering pipeline
- Complete bitwise lowering pipeline
- Complete comparison lowering pipeline

#### Multi-Dialect Integration
- Same operation lowered to multiple dialects
- Cross-dialect pattern registration
- Dialect-specific pattern selection

#### Control Flow Integration
- Control flow operations to CF dialect
- Structured control flow to SCF dialect

#### Memory Operations
- Memory allocation pipeline
- Load/store operations
- MemRef dialect integration

#### Function Operations
- Function calls
- Return statements
- Func dialect integration

#### Priority and Constraints
- Priority-based pattern selection
- Constraint-based pattern filtering
- Combined priority and constraint logic

#### Full Lowering Pipelines
- Complete arithmetic expression lowering: `(a + b) * c`
- Memory allocation, store, and load pipeline

### 3. Machine Pattern Tests

**File:** `tests/test_machine_patterns.cpp`

#### MLIR Arith Machine
- Basic machine properties (name, target language)
- Handling arithmetic operations
- Handling bitwise operations
- Handling comparison operations
- Pattern registration

#### MLIR CF Machine
- Control flow operation handling
- Pattern registration

#### MLIR SCF Machine
- Structured control flow handling
- Pattern registration

#### MLIR MemRef Machine
- Memory operation handling
- Pattern registration

#### MLIR Func Machine
- Function operation handling
- Pattern registration

#### Machine Registry
- Default machine registration
- Machine lookup by name
- Machine existence checking
- Available machines enumeration
- Custom machine registration
- Machine replacement

#### Integration Tests
- All machines register patterns
- Machine-specific pattern registration

### 4. Regression Tests

**File:** `regression-tests/test_pattern_lowering.sh`

#### Test Suites

1. **Pattern Matcher Unit Tests**
   - Runs all unit tests from test_pattern_matcher

2. **Pattern Integration Tests**
   - Runs all integration tests from test_pattern_integration

3. **Machine Pattern Tests**
   - Runs all machine tests from test_machine_patterns

4. **Pattern Database Tests**
   - Validates pattern database functionality

5. **YAML Pattern Loading**
   - Checks for pattern YAML files
   - Validates pattern file structure

6. **Pattern Coverage Analysis**
   - Analyzes pattern coverage
   - Checks for required headers

7. **Multi-Target Lowering Validation**
   - Validates TargetLanguage enum
   - Checks for MLIR dialect support

8. **Pattern Matcher API Validation**
   - Validates core API methods:
     - registerPattern
     - match
     - hasPattern
     - getPatternsForKind
     - getPatternCount
     - clear

9. **NodeKind Coverage**
   - Validates NodeKind categories
   - Checks for complete coverage

## Running Tests

### Building Tests

```bash
# From project root
cmake -B build
cmake --build build
```

### Running Unit Tests

```bash
# Run individual test suites
./build/test_pattern_matcher
./build/test_pattern_integration
./build/test_machine_patterns

# Run all tests via CTest
cd build
ctest
```

### Running Regression Tests

```bash
# From project root
./regression-tests/test_pattern_lowering.sh
```

## Test Metrics

### Coverage Statistics

- **Total Test Cases:** 100+
- **NodeKind Coverage:** All categories (CFG, Data, Arithmetic, Bitwise, Comparison, Float)
- **Target Languages:** 5 MLIR dialects (Arith, CF, SCF, MemRef, Func)
- **Pattern Types:** Simple, Priority-based, Constraint-based

### Quality Metrics

- **API Coverage:** All public methods tested
- **Edge Cases:** Null handling, empty inputs, boundary conditions
- **Integration:** Full pipeline validation
- **Regression:** Automated regression suite

## Test Organization

### Directory Structure

```
tests/
  ├── test_pattern_matcher.cpp       # Core unit tests
  ├── test_pattern_integration.cpp   # Integration tests
  ├── test_machine_patterns.cpp      # Machine tests
  └── CMakeLists.txt                 # Test build configuration

regression-tests/
  └── test_pattern_lowering.sh       # Regression test suite

docs/testing/
  └── PATTERN_MATCHER_TESTING.md     # This file
```

## Adding New Tests

### Adding a Unit Test

```cpp
TEST_F(PatternMatcherTest, MyNewTest) {
    // Register pattern
    matcher->registerPattern(
        NodeKind::MY_OP,
        TargetLanguage::MLIR_ARITH,
        [](Node* n) { return "my.op"; },
        10
    );

    // Verify
    EXPECT_TRUE(matcher->hasPattern(NodeKind::MY_OP, TargetLanguage::MLIR_ARITH));
}
```

### Adding an Integration Test

```cpp
TEST_F(PatternIntegrationTest, MyPipeline) {
    // Register pipeline patterns
    matcher->registerPattern(NodeKind::OP1, ...);
    matcher->registerPattern(NodeKind::OP2, ...);

    // Verify pipeline
    EXPECT_EQ(matcher->getPatternCount(), 2);
}
```

### Adding a Regression Test

Edit `regression-tests/test_pattern_lowering.sh`:

```bash
echo "Test Suite X: My New Tests"
echo "---------------------------"

my_test_function() {
    # Test logic here
    return 0
}

run_test "My Test Name" "my_test_function"
```

## Continuous Integration

### CI Pipeline

1. Build all tests
2. Run unit tests
3. Run integration tests
4. Run regression tests
5. Generate coverage report

### Required Checks

- All unit tests pass
- All integration tests pass
- Regression suite passes
- No memory leaks (valgrind)
- Code coverage > 80%

## Debugging Tests

### Running with Verbose Output

```bash
# GTest verbose output
./build/test_pattern_matcher --gtest_verbose

# Regression tests with debug
VERBOSE=1 ./regression-tests/test_pattern_lowering.sh
```

### Running Specific Tests

```bash
# Run specific test case
./build/test_pattern_matcher --gtest_filter=PatternMatcherTest.RegisterSimplePattern

# Run specific test suite
./build/test_pattern_matcher --gtest_filter=PatternMatcherTest.*
```

### Memory Leak Detection

```bash
# Run with valgrind
valgrind --leak-check=full ./build/test_pattern_matcher
```

## Known Limitations

1. **Mock Nodes:** Some tests require mock node infrastructure for complete validation
2. **Pattern Helpers:** Binary/unary operator helper tests need node creation support
3. **Match Function:** Full match() testing requires node instances

## Future Enhancements

1. Add benchmarking tests for pattern matching performance
2. Implement property-based testing for pattern transformations
3. Add fuzzing tests for pattern matcher robustness
4. Create visual regression tests for code generation output
5. Add stress tests for large pattern sets

## Contributing

When adding new patterns or features:

1. Add corresponding unit tests
2. Add integration tests for new pipelines
3. Update regression test suite
4. Document new test coverage
5. Ensure all tests pass before committing

## References

- [Pattern Matcher Design](../architecture/nway-induction-enum-strategy.md)
- [Machine Abstraction](../DEVELOPER_HANDOFF_BAND5.md)
- [Sea of Nodes IR](../architecture/sea-of-nodes-overview.md)
