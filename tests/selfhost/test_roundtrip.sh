#!/bin/bash
# test_roundtrip.sh - Full parse → codegen → compile → execute roundtrip
# Usage: test_roundtrip.sh <cppfort_binary> <input.cpp2> <output_binary>
set -euo pipefail

CPPFORT="$1"
INPUT="$2"
OUTPUT="$3"
TEMP_CPP="/tmp/cppfort_roundtrip_$$.cpp"

# Step 1: Parse and generate C++
"$CPPFORT" -c "$INPUT" > "$TEMP_CPP" 2>/dev/null

# Step 2: Compile generated C++
clang++ -std=c++20 -o "$OUTPUT" "$TEMP_CPP" 2>/dev/null

# Step 3: Execute and capture output
OUTPUT_TEXT=$("$OUTPUT" 2>/dev/null)

# Step 4: Verify output contains expected tag values
if echo "$OUTPUT_TEXT" | grep -q "join_tag"; then
    echo "ROUNDTRIP PASS: bootstrap_tags roundtrip successful"
    echo "Output: $OUTPUT_TEXT"
    exit 0
else
    echo "ROUNDTRIP FAIL: expected tag output not found"
    echo "Got: $OUTPUT_TEXT"
    exit 1
fi
