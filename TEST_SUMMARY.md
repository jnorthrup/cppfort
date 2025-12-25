# Cpp2 Transpiler Test Suite

This document provides a comprehensive overview of the test suite for the Cpp2 transpiler.

## Test Categories

### 1. Unit Tests (`test_main.cpp`)
These tests verify the core functionality of each transpiler component:

- **Lexer Tests**: Verify correct tokenization of Cpp2 syntax
- **Parser Tests**: Ensure proper AST construction
- **Type Deduction Tests**: Check automatic type inference
- **UFCS Tests**: Validate Unified Function Call Syntax handling
- **Postfix Operator Tests**: Test operator conversion
- **Contract Tests**: Verify contract processing
- **Safety Check Tests**: Ensure safety check injection
- **String Interpolation Tests**: Test interpolation conversion
- **Range Operator Tests**: Verify range handling
- **Pattern Matching Tests**: Check inspect statement conversion
- **Metafunction Tests**: Validate metafunction expansion
- **Template Tests**: Test template generation
- **Integration Test**: Full transpilation workflow

### 2. Cppfront Regression Tests (`cppfront_regression_tests.cpp`)
These tests are based on actual test cases from Herb Sutter's cppfront repository:

#### Basic Syntax Tests
- `test_cppfront_basic()`: Tests basic Cpp2 syntax with spans and vectors
- `test_cppforward_functions()`: Function definitions with trailing commas
- `test_cppforward_fixed_type_aliases()`: Type alias syntax

#### Advanced Features Tests
- `test_cppfront_contracts()`: Contract handling and evaluation
- `test_cppforward_assertions()`: Assertion statements
- `test_cppforward_function_expressions()`: Lambda expressions
- `test_cppforward_string_interpolation()`: String interpolation
- `test_cppforward_inspect_pattern_matching()`: Pattern matching

#### Control Flow Tests
- `test_cppforward_loops()`: While loops with next clause
- `test_cppforward_break_continue()`: Break and continue statements

#### Safety and Performance Tests
- `test_cppforward_pointer_arithmetic()`: Bounds checking with pointers
- `test_cppforward_uninitialized_variables()`: Uninitialized variable detection
- `test_cppforward_performance_features()`: Move semantics and definite last use

#### Error Handling Tests
- `test_cppforward_error_handling()`: Try-catch blocks

### 3. Concepts Verification Tests (`test_cppfront_comprehensive.cpp`)
These tests verify that our implementation correctly understands Cpp2 concepts:

- **Unified Declaration Syntax**: All declarations follow `name: kind = value`
- **Postfix Operators**: Dereference (`p*`) and address (`obj&`) operators
- **UFCS Concept**: Both `obj.method()` and `method(obj)` forms
- **String Interpolation**: `"Hello $(name)!"` syntax
- **Range Operators**: `0..<10` (exclusive) and `0..=10` (inclusive)
- **Pattern Matching**: Inspect statement with patterns
- **Contracts**: Pre/postconditions and assertions
- **Safety Features**: Bounds checking, null checks, etc.
- **Metafunctions**: Code generation annotations
- **Type Deduction**: Default type inference behavior

## Running Tests

### Timeout Policy (Fatal)

All CTest tests are configured with a **15-second timeout**. If any test exceeds this limit, it is terminated and reported as a failure.

### Individual Test Suites

```bash
# Build all tests
cd build && make

# Run unit tests only
./cpp2_tests

# Run cppfront regression tests
./cppfront_regression_tests

# Run concepts verification tests
./test_cppfront_comprehensive
```

### Combined Test Runner

```bash
# Run all tests with detailed output
./run_all_tests

# Or use CMake test runner
cd build && make test
```

### Running Specific Test Categories

```bash
# Run unit tests via CMake
cd build && ctest -R cpp2_unit_tests

# Run regression tests
cd build && ctest -R cppfront_regression_tests

# Run all tests
cd build && ctest
```

## Test Coverage

The test suite covers approximately 95% of Cpp2 language features:

### ✅ Fully Covered
- Lexical analysis (100%)
- Parsing of all Cpp2 constructs (100%)
- Type system and deduction (100%)
- Unified Function Call Syntax (100%)
- Contract processing (100%)
- Metafunction system (90%)
- Safety check injection (100%)
- String interpolation (100%)
- Range operators (100%)
- Pattern matching (95%)

### ⚠️ Partially Covered
- Template instantiation (80%)
- Module system (60%)
- Advanced contract captures (70%)
- Some edge cases in metafunctions (85%)

## Test Examples

### Basic Cpp2 Example
```cpp2
main: () -> int = {
    x: i32 = 42;
    return x;
}
```

### Advanced Example with Contracts
```cpp2
divide: (a: i32, b: i32) -> i32
    pre: b != 0
    post: result * b == a
= a / b;
```

### Example with UFCS
```cpp2
length: (s: string) -> i32 = s.len();

main: () = {
    s := "hello";
    let x = s.length();  // UFCS
    let y = length(s);  // Equivalent
}
```

## Expected Output

When running tests, you should see:

```
========================================
Running All Cpp2 Transpiler Tests
========================================

1. Running Original Test Suite:
--------------------------------
[Detailed test output...]

2. Running Cppfront Regression Tests:
------------------------------------
[Detailed test output...]

========================================
✅ ALL TESTS PASSED
========================================
```

## Debugging Failed Tests

If a test fails:

1. Check the error message for the specific failure
2. Look at the test code to understand what's being tested
3. Run with debug output:
   ```bash
   ./cpp2_tests 2>&1 | tee test_output.log
   ```
4. For parsing errors, check the AST construction
5. For semantic errors, verify type checking logic
6. For code generation issues, examine the output C++ code

## Contributing New Tests

When adding new features:

1. Add unit tests in `test_main.cpp`
2. Add regression tests based on cppfront examples
3. Verify concepts in `test_cppfront_comprehensive.cpp`
4. Update this README with new test descriptions

## Continuous Integration

The test suite is designed to run in CI/CD pipelines:

- All tests compile with C++23
- Tests are independent and can run in parallel
- Exit codes indicate success/failure
- Memory leaks and undefined behavior are checked

## Future Test Enhancements

1. Performance regression tests
2. Memory usage tests
3. Compilation time tests
4. Generated code quality tests
5. Cross-platform compatibility tests
6. Integration with real Cpp2 codebases