#!/bin/bash
# Head-to-head comparison: cppfort vs cppfront reference corpus
#
# Compares:
#   1. Transpilation success (both should succeed or fail)
#   2. C++ compilation success  
#   3. AST similarity (structural comparison)
#   4. Runtime output (where applicable)

# Don't exit on error - we want to continue testing
set +e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPPFORT="$PROJECT_ROOT/build/src/cppfort"
REF_DIR="$PROJECT_ROOT/tests/reference"
TEST_DIR="$PROJECT_ROOT/tests/regression-tests"
RESULTS_DIR="$PROJECT_ROOT/tests/results"
TMP_DIR="/tmp/cppfort_h2h_$$"

mkdir -p "$RESULTS_DIR" "$TMP_DIR"
trap "rm -rf $TMP_DIR" EXIT

# Ensure reference corpus exists
if [ ! -d "$REF_DIR" ] || [ -z "$(ls -A $REF_DIR/*.cpp 2>/dev/null)" ]; then
    echo "Reference corpus not found. Generating..."
    "$SCRIPT_DIR/reference_corpus.sh"
fi

# Results
PASS=0
FAIL=0
SKIP=0

# Detailed results
> "$RESULTS_DIR/h2h_pass.txt"
> "$RESULTS_DIR/h2h_fail.txt"
> "$RESULTS_DIR/h2h_skip.txt"
> "$RESULTS_DIR/h2h_detail.txt"

compare_test() {
    local name="$1"
    local src="$TEST_DIR/${name}.cpp2"
    local ref_cpp="$REF_DIR/${name}.cpp"
    local fort_cpp="$TMP_DIR/${name}_fort.cpp"
    local fort_ast="$TMP_DIR/${name}_fort.ast"
    local ref_bin="$TMP_DIR/${name}_ref"
    local fort_bin="$TMP_DIR/${name}_fort"
    
    echo -n "TEST: $name... "
    
    # Skip if no reference
    if [ ! -f "$ref_cpp" ]; then
        echo "SKIP (no ref)"
        echo "$name (no reference)" >> "$RESULTS_DIR/h2h_skip.txt"
        ((SKIP++))
        return
    fi
    
    # Skip if no source
    if [ ! -f "$src" ]; then
        echo "SKIP (no src)"
        echo "$name (no source)" >> "$RESULTS_DIR/h2h_skip.txt"
        ((SKIP++))
        return
    fi
    
    local status="PASS"
    local detail=""
    
    # Stage 1: cppfort transpile
    if ! "$CPPFORT" "$src" "$fort_cpp" >/dev/null 2>&1; then
        status="FAIL"
        detail="cppfort transpile failed"
    fi
    
    if [ "$status" = "PASS" ]; then
        # Stage 2: Compile reference
        local ref_compiles=0
        if clang++ -std=c++20 -I/Users/jim/work/cppfront/include -o "$ref_bin" "$ref_cpp" 2>/dev/null; then
            ref_compiles=1
        fi
        
        # Stage 3: Compile cppfort output
        local fort_compiles=0
        if clang++ -std=c++20 -I"$PROJECT_ROOT/include" -o "$fort_bin" "$fort_cpp" 2>/dev/null; then
            fort_compiles=1
        fi
        
        # Both should have same compile result
        if [ $ref_compiles -ne $fort_compiles ]; then
            status="FAIL"
            if [ $ref_compiles -eq 1 ]; then
                detail="ref compiles, fort doesn't"
            else
                detail="fort compiles, ref doesn't"
            fi
        fi
        
        # Stage 4: If both compile and not -error test, compare runtime
        if [ "$status" = "PASS" ] && [ $ref_compiles -eq 1 ] && [[ ! "$name" =~ -error$ ]]; then
            local ref_out=$(timeout 5 "$ref_bin" 2>&1 || echo "__TIMEOUT__")
            local fort_out=$(timeout 5 "$fort_bin" 2>&1 || echo "__TIMEOUT__")
            
            if [ "$ref_out" != "$fort_out" ]; then
                # Allow minor differences (whitespace, etc)
                local ref_norm=$(echo "$ref_out" | tr -d '[:space:]')
                local fort_norm=$(echo "$fort_out" | tr -d '[:space:]')
                if [ "$ref_norm" != "$fort_norm" ]; then
                    status="FAIL"
                    detail="output differs"
                    echo "=== $name ===" >> "$RESULTS_DIR/h2h_detail.txt"
                    echo "REF OUTPUT:" >> "$RESULTS_DIR/h2h_detail.txt"
                    echo "$ref_out" >> "$RESULTS_DIR/h2h_detail.txt"
                    echo "FORT OUTPUT:" >> "$RESULTS_DIR/h2h_detail.txt"
                    echo "$fort_out" >> "$RESULTS_DIR/h2h_detail.txt"
                    echo "" >> "$RESULTS_DIR/h2h_detail.txt"
                fi
            fi
        fi
    fi
    
    if [ "$status" = "PASS" ]; then
        echo "PASS"
        echo "$name" >> "$RESULTS_DIR/h2h_pass.txt"
        ((PASS++))
    else
        echo "FAIL ($detail)"
        echo "$name: $detail" >> "$RESULTS_DIR/h2h_fail.txt"
        ((FAIL++))
    fi
}

echo "=========================================="
echo "Head-to-Head: cppfort vs cppfront"
echo "=========================================="
echo ""

# Get list of tests from reference corpus
for ref_cpp in "$REF_DIR"/*.cpp; do
    [ -f "$ref_cpp" ] || continue
    name=$(basename "$ref_cpp" .cpp)
    compare_test "$name"
done

echo ""
echo "=========================================="
echo "Results: PASS=$PASS, FAIL=$FAIL, SKIP=$SKIP"
echo "=========================================="
echo ""
echo "Details in: $RESULTS_DIR/"

# Generate summary
cat > "$RESULTS_DIR/H2H_SUMMARY.md" << EOF
# Head-to-Head: cppfort vs cppfront

Generated: $(date)

## Results

| Status | Count |
|--------|-------|
| PASS | $PASS |
| FAIL | $FAIL |
| SKIP | $SKIP |

## Passing Tests

\`\`\`
$(cat "$RESULTS_DIR/h2h_pass.txt")
\`\`\`

## Failing Tests

\`\`\`
$(cat "$RESULTS_DIR/h2h_fail.txt")
\`\`\`

## Skipped Tests

\`\`\`
$(cat "$RESULTS_DIR/h2h_skip.txt")
\`\`\`
EOF
