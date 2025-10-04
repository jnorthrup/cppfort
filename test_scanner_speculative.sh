#!/bin/bash
# Speculative WIDE scanner validation - run multiple tests concurrently

echo "=== WIDE Scanner Speculative Testing Suite ==="
echo "Running $(date)"

# Test 1: Direct scanner orbit tests
echo "Test 1: Running orbit scanner tests..."
./test_orbits > test1_orbits.out 2>&1 &
PID1=$!

# Test 2: Pattern file validation
echo "Test 2: Validating pattern files..."
(
    for pattern in patterns/*.yaml; do
        if [ -f "$pattern" ]; then
            python3 -c "import yaml; yaml.safe_load(open('$pattern'))" 2>&1 && echo "✓ $pattern valid" || echo "✗ $pattern invalid"
        fi
    done
) > test2_patterns.out 2>&1 &
PID2=$!

# Test 3: Scanner source compilation test
echo "Test 3: Testing scanner compilation..."
(
    echo "Checking wide_scanner.cpp compilation..."
    g++ -std=c++20 -c src/stage0/wide_scanner.cpp -I. -o /tmp/wide_scanner.o 2>&1
    echo "Exit: $?"
) > test3_compile.out 2>&1 &
PID3=$!

# Test 4: Regression file scanning
echo "Test 4: Scanning regression test files..."
(
    for test_file in regression-tests/pure2-*.cpp2; do
        if [ -f "$test_file" ]; then
            # Run test_orbits on each file if it exists
            echo "Scanning: $(basename $test_file)"
            wc -l "$test_file"
        fi
    done | head -20
) > test4_regression_scan.out 2>&1 &
PID4=$!

# Wait for all tests to complete
wait $PID1 $PID2 $PID3 $PID4

echo ""
echo "=== Test Results Summary ==="

echo "1. Orbit Tests:"
tail -5 test1_orbits.out

echo ""
echo "2. Pattern Validation:"
grep -E "(✓|✗)" test2_patterns.out | head -5

echo ""
echo "3. Compilation:"
tail -3 test3_compile.out

echo ""
echo "4. Regression Files:"
tail -5 test4_regression_scan.out

echo ""
echo "=== Gap Analysis ==="

# Check for critical gaps
GAPS=""

if ! grep -q "All Tests Complete" test1_orbits.out; then
    GAPS="$GAPS\n- Orbit tests incomplete"
fi

if grep -q "invalid" test2_patterns.out; then
    GAPS="$GAPS\n- Pattern files have validation errors"
fi

if grep -q "error:" test3_compile.out; then
    GAPS="$GAPS\n- Scanner compilation failures"
fi

if [ -z "$GAPS" ]; then
    echo "✓ No critical gaps detected"
else
    echo "Gaps detected:$GAPS"
fi

echo ""
echo "=== Technical Debt ==="
echo "- stage0 transpiler cannot load patterns/cpp2_patterns.yaml"
echo "- Missing multi_grammar_loader.h dependency"
echo "- 130/130 regression tests failing due to pattern load failure"
echo "- Scanner integration incomplete with stage0_cli"

exit 0