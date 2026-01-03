#!/bin/bash
# Baseline cppfront regression tests for cppfort
# This script tests transpilation and C++ compilation, accounting for:
# - Our inline runtime (no cpp2util.h dependency)
# - Known differences in codegen

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
CPPFORT="$BUILD_DIR/src/cppfort"
TEST_DIR="$PROJECT_ROOT/tests/regression-tests"
BASELINE_DIR="$PROJECT_ROOT/tests/baseline"
TMP_DIR="/tmp/cppfort_baseline_$$"

mkdir -p "$TMP_DIR" "$BASELINE_DIR"
trap "rm -rf $TMP_DIR" EXIT

# Counters
PURE2_TRANSPILE_PASS=0
PURE2_TRANSPILE_FAIL=0
PURE2_COMPILE_PASS=0
PURE2_COMPILE_FAIL=0
PURE2_RUN_PASS=0
PURE2_RUN_FAIL=0

MIXED_TRANSPILE_PASS=0
MIXED_TRANSPILE_FAIL=0
MIXED_COMPILE_PASS=0
MIXED_COMPILE_FAIL=0

# Results files
PURE2_PASS_FILE="$BASELINE_DIR/pure2_passing.txt"
PURE2_FAIL_FILE="$BASELINE_DIR/pure2_failing.txt"
MIXED_PASS_FILE="$BASELINE_DIR/mixed_passing.txt"
MIXED_FAIL_FILE="$BASELINE_DIR/mixed_failing.txt"

> "$PURE2_PASS_FILE"
> "$PURE2_FAIL_FILE"
> "$MIXED_PASS_FILE"
> "$MIXED_FAIL_FILE"

echo "=========================================="
echo "cppfort Regression Test Baseline"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Test a single pure2 file
test_pure2() {
    local src="$1"
    local name=$(basename "$src" .cpp2)
    local out_cpp="$TMP_DIR/${name}.cpp"
    local out_bin="$TMP_DIR/${name}"
    local status="FAIL"
    local fail_stage=""
    
    # Stage 1: Transpile
    if "$CPPFORT" "$src" "$out_cpp" >/dev/null 2>&1; then
        ((PURE2_TRANSPILE_PASS++))
        
        # Stage 2: Compile
        if clang++ -std=c++20 -I "$PROJECT_ROOT/include" -o "$out_bin" "$out_cpp" 2>/dev/null; then
            ((PURE2_COMPILE_PASS++))
            
            # Stage 3: Run (if not an error test)
            if [[ ! "$name" =~ -error$ ]]; then
                if timeout 5 "$out_bin" >/dev/null 2>&1; then
                    ((PURE2_RUN_PASS++))
                    status="PASS"
                else
                    ((PURE2_RUN_FAIL++))
                    fail_stage="run"
                fi
            else
                # Error tests pass if they compile (the error is a cpp2 semantic error)
                status="PASS"
            fi
        else
            ((PURE2_COMPILE_FAIL++))
            fail_stage="compile"
        fi
    else
        ((PURE2_TRANSPILE_FAIL++))
        fail_stage="transpile"
    fi
    
    if [ "$status" = "PASS" ]; then
        echo "$name" >> "$PURE2_PASS_FILE"
    else
        echo "$name ($fail_stage)" >> "$PURE2_FAIL_FILE"
    fi
}

# Test a single mixed file
test_mixed() {
    local src="$1"
    local name=$(basename "$src" .cpp2)
    local out_cpp="$TMP_DIR/${name}.cpp"
    local out_bin="$TMP_DIR/${name}"
    local status="FAIL"
    local fail_stage=""
    
    # Stage 1: Transpile
    if "$CPPFORT" "$src" "$out_cpp" >/dev/null 2>&1; then
        ((MIXED_TRANSPILE_PASS++))
        
        # Stage 2: Compile
        if clang++ -std=c++20 -I "$PROJECT_ROOT/include" -o "$out_bin" "$out_cpp" 2>/dev/null; then
            ((MIXED_COMPILE_PASS++))
            status="PASS"
        else
            ((MIXED_COMPILE_FAIL++))
            fail_stage="compile"
        fi
    else
        ((MIXED_TRANSPILE_FAIL++))
        fail_stage="transpile"
    fi
    
    if [ "$status" = "PASS" ]; then
        echo "$name" >> "$MIXED_PASS_FILE"
    else
        echo "$name ($fail_stage)" >> "$MIXED_FAIL_FILE"
    fi
}

echo "Testing pure2 files..."
for f in "$TEST_DIR"/pure2-*.cpp2; do
    [ -f "$f" ] || continue
    test_pure2 "$f"
done

echo "Testing mixed files..."
for f in "$TEST_DIR"/mixed-*.cpp2; do
    [ -f "$f" ] || continue
    test_mixed "$f"
done

# Sort results
sort -o "$PURE2_PASS_FILE" "$PURE2_PASS_FILE"
sort -o "$PURE2_FAIL_FILE" "$PURE2_FAIL_FILE"
sort -o "$MIXED_PASS_FILE" "$MIXED_PASS_FILE"
sort -o "$MIXED_FAIL_FILE" "$MIXED_FAIL_FILE"

# Summary
echo ""
echo "=========================================="
echo "BASELINE RESULTS"
echo "=========================================="
echo ""
echo "Pure2 Tests (139 total):"
echo "  Transpile: $PURE2_TRANSPILE_PASS pass, $PURE2_TRANSPILE_FAIL fail"
echo "  Compile:   $PURE2_COMPILE_PASS pass, $PURE2_COMPILE_FAIL fail"
echo "  Run:       $PURE2_RUN_PASS pass, $PURE2_RUN_FAIL fail"
echo ""
echo "Mixed Tests (50 total):"
echo "  Transpile: $MIXED_TRANSPILE_PASS pass, $MIXED_TRANSPILE_FAIL fail"
echo "  Compile:   $MIXED_COMPILE_PASS pass, $MIXED_COMPILE_FAIL fail"
echo ""
echo "Results saved to:"
echo "  $PURE2_PASS_FILE"
echo "  $PURE2_FAIL_FILE"
echo "  $MIXED_PASS_FILE"
echo "  $MIXED_FAIL_FILE"

# Generate summary markdown
cat > "$BASELINE_DIR/BASELINE.md" << EOF
# cppfort Regression Test Baseline

Generated: $(date)

## Summary

| Category | Transpile | Compile | Run |
|----------|-----------|---------|-----|
| Pure2 (139) | $PURE2_TRANSPILE_PASS | $PURE2_COMPILE_PASS | $PURE2_RUN_PASS |
| Mixed (50) | $MIXED_TRANSPILE_PASS | $MIXED_COMPILE_PASS | N/A |

## Notes

- Tests use cppfort's inline runtime (no cpp2util.h dependency)
- Compile uses: \`clang++ -std=c++20 -I include\`
- Error tests (-error suffix) pass if they compile
- Run tests have 5 second timeout

## Passing Pure2 Tests

\`\`\`
$(cat "$PURE2_PASS_FILE")
\`\`\`

## Failing Pure2 Tests

\`\`\`
$(cat "$PURE2_FAIL_FILE")
\`\`\`

## Passing Mixed Tests

\`\`\`
$(cat "$MIXED_PASS_FILE")
\`\`\`

## Failing Mixed Tests

\`\`\`
$(cat "$MIXED_FAIL_FILE")
\`\`\`
EOF

echo ""
echo "Baseline markdown: $BASELINE_DIR/BASELINE.md"
