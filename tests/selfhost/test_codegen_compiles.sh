#!/bin/bash
# test_codegen_compiles.sh - Verify generated C++ compiles without errors
# Usage: test_codegen_compiles.sh <cppfort_binary> <input.cpp2>
set -euo pipefail

CPPFORT="$1"
INPUT="$2"
TEMP_CPP="/tmp/cppfort_codegen_$$.cpp"

# Step 1: Generate C++
"$CPPFORT" -c "$INPUT" > "$TEMP_CPP" 2>/dev/null

# Step 2: Try to compile
if clang++ -std=c++20 -fsyntax-only "$TEMP_CPP" 2>/dev/null; then
    echo "CODEGEN COMPILES PASS"
    exit 0
else
    echo "CODEGEN COMPILES FAIL: generated C++ does not compile"
    echo "Generated code:"
    cat "$TEMP_CPP"
    exit 1
fi
