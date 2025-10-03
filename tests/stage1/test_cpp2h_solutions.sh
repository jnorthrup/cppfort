#!/bin/bash

# test_cpp2h_solutions.sh - Test the different cpp2.h distribution solutions

set -e

echo "Testing Stage 1 cpp2.h distribution solutions..."

# Create a simple test file
TEST_INPUT="/tmp/test.cpp2"
TEST_OUTPUT="/tmp/test.cpp"

cat > "$TEST_INPUT" << 'EOF'
main: () -> int = {
    std::cout << "Hello, World!\n";
    return 0;
}
EOF

# Create a more complex test file that uses cpp2 features
COMPLEX_INPUT="/tmp/complex.cpp2"
COMPLEX_OUTPUT="/tmp/complex.cpp"

cat > "$COMPLEX_INPUT" << 'EOF'
// Test file that uses cpp2.h features
process_copy: (copy x: int) -> int = {
    return x * 2;
}

process_in: (in x: int) -> int = {
    return x + 1;
}

main: () -> int = {
    x: int = 42;
    y: int = process_copy(x);
    z: int = process_in(x);
    std::cout << "x=" << x << " y=" << y << " z=" << z << "\n";
    return 0;
}
EOF

echo "Created test input files"

# Build the transpiler if needed
if [ ! -f "build/src/stage1/transpiler" ]; then
    echo "Building Stage 1 transpiler..."
    cd build && make stage1_transpiler && cd ..
fi

# Test 1: Default behavior (should include cpp2.h)
echo "Test 1: Default behavior (includes cpp2.h)"
./build/src/stage1/transpiler "$TEST_INPUT" "$TEST_OUTPUT"
if grep -q '#include "cpp2.h"' "$TEST_OUTPUT"; then
    echo "  PASS: Generated code includes cpp2.h"
else
    echo "  FAIL: Generated code does not include cpp2.h"
    echo "  Generated code:"
    cat "$TEST_OUTPUT"
fi

# Test 2: Inline cpp2.h contents
echo "Test 2: Inline cpp2.h contents"
./build/src/stage1/transpiler "$TEST_INPUT" "$TEST_OUTPUT" --inline-cpp2
if grep -q '#include "cpp2.h"' "$TEST_OUTPUT"; then
    echo "  FAIL: Generated code still includes cpp2.h when inlining"
elif grep -q "namespace cpp2 {" "$TEST_OUTPUT" ; then
    echo "  PASS: Generated code inlines cpp2.h contents"
else
    echo "  FAIL: Generated code does not inline cpp2.h contents"
    echo "  Generated code:"
    cat "$TEST_OUTPUT"
fi

# Test 3: Bundle cpp2.h contents
echo "Test 3: Bundle cpp2.h contents"
./build/src/stage1/transpiler "$TEST_INPUT" "$TEST_OUTPUT" --bundle-cpp2
if head -20 "$TEST_OUTPUT" | grep -q "Bundled cpp2.h"; then
    echo "  PASS: Generated code bundles cpp2.h contents"
else
    echo "  FAIL: Generated code does not bundle cpp2.h contents"
    echo "  Generated code (first 30 lines):"
    head -30 "$TEST_OUTPUT"
fi

# Test 4: Compile the generated code
echo "Test 4: Compile generated code with default settings"
./build/src/stage1/transpiler "$TEST_INPUT" "$TEST_OUTPUT"
if g++ -I./include "$TEST_OUTPUT" -o /tmp/test_default 2>/dev/null; then
    echo "  PASS: Generated code compiles with -I./include"
else
    echo "  FAIL: Generated code does not compile with -I./include"
fi

# Test 5: Compile the generated code with inlined header
echo "Test 5: Compile generated code with inlined cpp2.h"
./build/src/stage1/transpiler "$TEST_INPUT" "$TEST_OUTPUT" --inline-cpp2
if g++ "$TEST_OUTPUT" -o /tmp/test_inline 2>/dev/null; then
    echo "  PASS: Generated code with inlined cpp2.h compiles without -I flags"
else
    echo "  FAIL: Generated code with inlined cpp2.h does not compile without -I flags"
fi

# Test 6: Compile the generated code with bundled header
echo "Test 6: Compile generated code with bundled cpp2.h"
./build/src/stage1/transpiler "$TEST_INPUT" "$TEST_OUTPUT" --bundle-cpp2
if g++ "$TEST_OUTPUT" -o /tmp/test_bundle 2>/dev/null; then
    echo "  PASS: Generated code with bundled cpp2.h compiles without -I flags"
else
    echo "  FAIL: Generated code with bundled cpp2.h does not compile without -I flags"
fi

# Test 7: Complex example with actual cpp2.h usage
echo "Test 7: Complex example with actual cpp2.h usage"
./build/src/stage1/transpiler "$COMPLEX_INPUT" "$COMPLEX_OUTPUT" --inline-cpp2
if g++ "$COMPLEX_OUTPUT" -o /tmp/complex_inline 2>/dev/null; then
    echo "  PASS: Complex code with inlined cpp2.h compiles without -I flags"
    # Run the program to make sure it works
    if /tmp/complex_inline | grep -q "x=42 y=84 z=43"; then
        echo "  PASS: Complex program produces expected output"
    else
        echo "  FAIL: Complex program does not produce expected output"
    fi
else
    echo "  FAIL: Complex code with inlined cpp2.h does not compile without -I flags"
fi

echo "Tests completed."