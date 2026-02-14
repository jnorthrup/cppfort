#!/usr/bin/env bash
# test_one.sh - Single test runner for cppfort
# Usage: ./test_one.sh <test_name> [--verbose]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPPFORT_BIN="$PROJECT_ROOT/build/cppfort"
CORPUS_DIR="$PROJECT_ROOT/third_party/cppfront/regression-tests"
WORK_DIR="$PROJECT_ROOT/build_clean/test_work"

mkdir -p "$WORK_DIR"

test_name="$1"
verbose="${2:-""}"

# Find test file
test_file=""
if [ -f "$test_name" ]; then
    test_file="$test_name"
elif [ -f "$CORPUS_DIR/$test_name.cpp2" ]; then
    test_file="$CORPUS_DIR/$test_name.cpp2"
elif [ -f "$CORPUS_DIR/$test_name" ]; then
    test_file="$CORPUS_DIR/$test_name"
else
    echo "ERROR: Test file not found: $test_name"
    exit 1
fi

# Output paths
basename_test=$(basename "$test_file" .cpp2)
cpp_file="$WORK_DIR/${basename_test}.cpp"
log_file="$WORK_DIR/${basename_test}.log"

echo "Testing: $basename_test"
echo "Input: $test_file"

# Transpile
if [ -n "$verbose" ]; then
    echo "--- Transpilation ---"
    "$CPPFORT_BIN" "$test_file" "$cpp_file" 2>&1 | tee "$log_file" || true
else
    "$CPPFORT_BIN" "$test_file" "$cpp_file" > "$log_file" 2>&1 || true
fi

if [ ! -f "$cpp_file" ]; then
    echo "FAIL: Transpilation failed (no output file)"
    cat "$log_file"
    exit 1
fi

# Show output if verbose
if [ -n "$verbose" ]; then
    echo "--- Generated C++ ---"
    head -30 "$cpp_file"
    echo "..."
fi

echo "✓ Transpilation succeeded"
echo "Output: $cpp_file"
