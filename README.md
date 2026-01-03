# Cpp2 Transpiler

A comprehensive C++23 source-to-source transpiler that converts Cpp2 code to modern C++ (C++20/23).

## Features

- **Full Cpp2 Language Support**: Implements all major Cpp2 features including unified declaration syntax, UFCS, contracts, metafunctions, and pattern matching
- **Safety by Default**: Injects runtime checks for bounds, null pointers, division by zero, and mixed-sign comparisons
- **Modern C++ Generation**: Produces idiomatic C++20/23 code with move semantics, concepts, and ranges
- **Contract System**: Full support for preconditions, postconditions, and assertions with customizable violation handlers
- **Metafunction System**: Compile-time code generation for common patterns (value semantics, ordering, copyability, etc.)
- **Template Support**: Full C++ template generation from Cpp2 template syntax
- **Based on C++23**: Uses modern C++23 features including spans, string_views, and std::format

## Architecture

The transpiler is built with a clean, modular architecture:

1. **Lexer**: Tokenizes Cpp2 source code
2. **Parser**: Builds an Abstract Syntax Tree (AST) from tokens
   - Grammar: `grammar/cpp2.combinators.md` (orthogonal combinator spec)
   - EBNF: `grammar/cpp2.ebnf` (formal grammar)
3. **Semantic Analyzer**: Performs type checking and symbol resolution
4. **Safety Checker**: Identifies potential safety issues and injection points
5. **Metafunction Processor**: Expands metafunction annotations
6. **Contract Processor**: Processes and generates contract checks
7. **Code Generator**: Generates C++20/23 code from the processed AST

## Building

### Prerequisites

- C++23 compliant compiler (GCC 13+, Clang 16+, MSVC 19.36+)
- CMake 3.28 or later
- Git

### Build Steps

```bash
git clone <repository-url>
cd cppfort
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Usage

```bash
# Transpile a Cpp2 file
./cpp2_transpiler input.cpp2 output.cpp

# Compile the generated C++ code
g++ -std=c++23 -O2 output.cpp -o output
```

## Cpp2 Language Features

### Unified Declaration Syntax

```cpp2
// Variables
let x: i32 = 42;          // Immutable
mut y: f64 = 3.14;        // Mutable
const z: string = "hello"; // Compile-time constant

// Functions
func add(a: i32, b: i32) -> i32 = a + b;

// Types
type Point = {
    x: f64,
    y: f64,
};
```

### Unified Function Call Syntax (UFCS)

```cpp2
func length(s: string) -> i32 = s.len();

// Both are equivalent
let len1 = "hello".length();
let len2 = length("hello");
```

### Postfix Operators

```cpp2
let p: i32* = &x;
let value = p*;     // Dereference
let addr = value&;  // Address
```

### Contracts

```cpp2
func divide(a: i32, b: i32) -> i32
    pre: b != 0
    post: result * b == a
{
    return a / b;
}
```

### Pattern Matching

```cpp2
inspect value {
    0 => "zero",
    1..=9 => "single digit",
    n if n < 0 => "negative",
    _ => "other",
}
```

### String Interpolation

```cpp2
let name = "World";
let message = "Hello, $(name)!";
```

### Range Operators

```cpp2
for i in 0..<10 {    // Exclusive range
    // ...
}

for i in 0..=10 {    // Inclusive range
    // ...
}
```

### Metafunctions

```cpp2
@value          // Generate value semantics
@ordered        // Generate comparison operators
@copyable       // Ensure copy operations
@hashable       // Generate hash function
type Person = {
    name: string,
    age: i32,
};
```

## Safety Features

The transpiler automatically injects safety checks:

- **Array bounds checking**: Validates array/subscript access
- **Null pointer checks**: Prevents null dereferencing
- **Division by zero**: Checks for zero divisors
- **Mixed-sign comparisons**: Warns about signed/unsigned comparisons
- **Use-after-move detection**: Identifies potential use-after-move scenarios

## Testing

Run the test suite:

```bash
cd build
make test
# Or
./cpp2_tests
```

## Examples

See the `examples/` directory for Cpp2 examples:

- `hello.cpp2`: Simple hello world
- `vector.cpp2`: Using arrays and UFCS
- `person.cpp2`: Metafunctions and contracts

## Project Structure

```
cppfort/
├── include/           # Header files
├── src/              # Source implementations
├── tests/            # Test suite
├── examples/         # Cpp2 examples
├── docs/             # Documentation
├── CMakeLists.txt     # Build configuration
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Based on the Cpp2 design by Herb Sutter and the ISO C++ committee.