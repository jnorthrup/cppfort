# Stage 1 Transpiler User Guide

## Overview

The Stage 1 transpiler converts Cpp2 source code to standard C++ that can be compiled with any C++17 or later compiler.

## Basic Usage

```bash
# Basic transpilation
./stage1 input.cpp2 output.cpp

# With legacy invocation style (used by regression tests)
./stage1 transpile input.cpp2 output.cpp
```

## Handling the cpp2.h Dependency

The generated C++ code may depend on the `cpp2.h` header file, which contains runtime support for Cpp2 language features. There are three ways to handle this dependency:

### Option 1: Default Behavior (Include External Header)

By default, the transpiler generates code that includes `cpp2.h`:

```bash
./stage1 input.cpp2 output.cpp
```

To compile the generated code, you need to specify the include path:

```bash
./stage1 input.cpp2 output.cpp
g++ -I./include output.cpp -o program
```

### Option 2: Inline cpp2.h Contents

Use the `--inline-cpp2` flag to embed the required cpp2.h definitions directly in the generated code:

```bash
./stage1 input.cpp2 output.cpp --inline-cpp2
g++ output.cpp -o program  # No -I flag needed
```

This creates a standalone C++ file that doesn't require external dependencies.

### Option 3: Bundle cpp2.h Contents

Use the `--bundle-cpp2` flag to prepend the complete cpp2.h contents at the beginning of the generated code:

```bash
./stage1 input.cpp2 output.cpp --bundle-cpp2
g++ output.cpp -o program  # No -I flag needed
```

This also creates a standalone C++ file that doesn't require external dependencies.

## Installation Script

For development environments, you can install the cpp2.h header to a standard location:

```bash
# Install to /usr/local/include
sudo ./scripts/install_stage1.sh

# Install to custom location
sudo ./scripts/install_stage1.sh --prefix=/opt/cppfort
```

To uninstall:

```bash
# Uninstall from /usr/local/include
sudo ./scripts/uninstall_stage1.sh

# Uninstall from custom location
sudo ./scripts/uninstall_stage1.sh --prefix=/opt/cppfort
```

After installation, you can compile generated code without specifying include paths:

```bash
./stage1 input.cpp2 output.cpp
g++ output.cpp -o program
```

## Recommendations

- For **production use**: Use `--inline-cpp2` or `--bundle-cpp2` to create standalone output
- For **development**: Use the installation script for convenience
- For **minimal generated code size**: Use the default behavior with `-I` flags

## Example

Create a simple Cpp2 program:

```cpp2
main: () -> int = {
    std::cout << "Hello, World!\n";
    return 0;
}
```

Transpile and compile:

```bash
# Method 1: Default with -I flag
./stage1 hello.cpp2 hello.cpp
g++ -I./include hello.cpp -o hello

# Method 2: Inline for standalone output
./stage1 hello.cpp2 hello.cpp --inline-cpp2
g++ hello.cpp -o hello

# Method 3: Bundle for standalone output
./stage1 hello.cpp2 hello.cpp --bundle-cpp2
g++ hello.cpp -o hello
```