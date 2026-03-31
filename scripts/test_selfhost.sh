#!/bin/bash
set -e

# Minimal Dogfood Test Script
# Tests that cppfort can compile bootstrap_tags.cpp2 to a working executable

echo "=== cppfort Minimal Dogfood Test ==="
echo ""

# Test 1: cppfort can parse source
echo "Test 1: Parse source to canonical AST"
./build/src/selfhost/cppfort src/selfhost/bootstrap_tags.cpp2 2>&1 | head -5

echo ""

# Test 2: cppfort generates C++ code
echo "Test 2: Generate C++ code"
./build/src/selfhost/cppfort src/selfhost/bootstrap_tags.cpp2 -c -o /tmp/cppfort_output.cpp 2>&1
echo "  Generated C++ code:"
head -20 /tmp/cppfort_output.cpp

echo ""

# Test 3: cppfort compiles to executable
echo "Test 3: Compile to executable"
./build/src/selfhost/cppfort src/selfhost/bootstrap_tags.cpp2 -o /tmp/cppfort_exe 2>&1

echo ""

# Test 4: Run the executable
echo "Test 4: Run compiled executable"
OUTPUT=$(/tmp/cppfort_exe 2>&1)
echo "  Output: $OUTPUT"

echo ""

# Test 5: Verify pipeline
echo "Test 5: Verification"
if [ -x /tmp/cppfort_exe ]; then
    echo "  Executable created: YES"
else
    echo "  Executable created: NO"
    exit 1
fi

echo ""
echo "=== SUCCESS: Minimal Dogfood Pipeline Works ==="
echo ""
echo "Pipeline: cpp2 source -> canonical AST -> C++ -> executable"
echo ""
echo "Note: Parser captures partial semantics. Full tag capture"
echo "      requires extended parser coverage (next phase)."