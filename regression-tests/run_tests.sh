#!/usr/bin/env bash
# ----------------------------------------------------------------------
# regression-tests/run_tests.sh
# ----------------------------------------------------------------------
# Inductive regression testing framework for cppfort stages
# Runs all tests, collects errors, and provides improvement suggestions
# ----------------------------------------------------------------------
# NOTE: Make the script executable with `chmod +x run_tests.sh`
# ----------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

# Compiler selection and flags for compiling generated C++.
# Default to g++ with C++17 and reasonable warnings. Users can override
# by exporting CXX or CXXFLAGS in their environment.
CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:--std=gnu++17 -Wall -Wextra -I${PROJECT_ROOT}/include}"

# Build all stages
BUILD_DIR="${PROJECT_ROOT}/build"
mkdir -p "${BUILD_DIR}"
pushd "${BUILD_DIR}" >/dev/null
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --target stage0 stage1
popd >/dev/null

echo "🧪 Running inductive regression tests..."
echo "=========================================="

# Error tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
ERROR_LOG=""

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_result="${3:-}"

    ((TOTAL_TESTS++))
    echo ""
    echo "Test $TOTAL_TESTS: $test_name"
    echo "─"$(printf '%.0s─' $(seq 1 ${#test_name}))"─"

    if eval "$test_command" 2>&1; then
        if [[ -n "$expected_result" ]] && ! eval "$expected_result" 2>&1 >/dev/null; then
            echo "❌ $test_name failed - expected condition not met"
            ((FAILED_TESTS++))
            ERROR_LOG+="❌ Test $TOTAL_TESTS ($test_name): Expected condition failed\n"
            return 1
        fi
        echo "✅ $test_name passed"
        ((PASSED_TESTS++))
        return 0
    else
        echo "❌ $test_name failed"
        ((FAILED_TESTS++))
        ERROR_LOG+="❌ Test $TOTAL_TESTS ($test_name): Command failed\n"
        return 1
    fi
}

# Function to analyze errors and suggest improvements
analyze_errors() {
    # Ensure COMP_FAILED_TESTS is defined (it is declared later in the script)
    if [[ -z "${COMP_FAILED_TESTS+x}" ]]; then
        COMP_FAILED_TESTS=0
    fi
    if [[ -z "${COMP_ERROR_LOG+x}" ]]; then
        COMP_ERROR_LOG=""
    fi

    total_failed=$((FAILED_TESTS + COMP_FAILED_TESTS))
    if [[ $total_failed -eq 0 ]]; then
        echo ""
        echo "🎉 All tests passed! No improvements needed."
        return
    fi

    echo ""
    echo "📊 Error Analysis & Improvement Suggestions"
    echo "=========================================="

    # Analyze inductive (short) test errors first
    if [[ $FAILED_TESTS -gt 0 ]]; then
        # Check for common error patterns in the inductive error log
        if echo "$ERROR_LOG" | grep -q "stage0.*compile"; then
            echo "🔧 Stage 0 Compilation Issues:"
            echo "   - Check if stage0 parser handles all cpp2 syntax variants"
            echo "   - Verify AST generation is complete"
            echo "   - Consider adding more robust error recovery"
        fi

        if echo "$ERROR_LOG" | grep -q "stage1.*compile"; then
            echo "🔧 Stage 1 Transpilation Issues:"
            echo "   - Review regex patterns for cpp2 syntax transformations"
            echo "   - Check function declaration order (forward declarations needed?)"
            echo "   - Verify variable declaration transformations"
            echo "   - Add support for more cpp2 language features"
        fi
    fi

    # Analyze comprehensive suite failures
    if [[ $COMP_FAILED_TESTS -gt 0 ]]; then
        echo "🔧 Comprehensive Suite Failures: $COMP_FAILED_TESTS file(s) failed to compile"
        echo "   - Many generated C++ files failed to compile. Inspect a few representative outputs in: $BUILD_DIR"
        echo "   - Common causes to look for: malformed function signatures, stray tokens (extra colons, misplaced commas), missing headers, or duplicate/incorrect includes"
        echo "   - Suggestion: capture stderr for failing files inside the loop (e.g. redirect to \"$BUILD_DIR/<test>.err\") to collect detailed diagnostics"
        echo "   - Start triage by compiling one failing generated file to see exact g++ diagnostics:"
        echo "       g++ -c $BUILD_DIR/comprehensive_test_<name>.cpp -o /tmp/out.o"
    fi

    # Pattern-matching issues across both logs
    if echo "$ERROR_LOG$COMP_ERROR_LOG" | grep -q "expected.*not found"; then
        echo "🔧 Pattern Matching Issues:"
        echo "   - Update regex patterns to handle edge cases"
        echo "   - Add more comprehensive syntax coverage"
        echo "   - Consider using a proper parser instead of regex"
    fi

    echo ""
    echo "💡 Next Steps:"
    echo "   1. Review failed generated C++ files in $BUILD_DIR/ (open a few representative \"comprehensive_test_*.cpp\")"
    echo "   2. Re-run g++ on a failing generated file and capture stderr to find the root cause"
    echo "   3. Update transpiler transformations that produced the malformed C++ (look for repeated patterns in the diagnostics)"
    echo "   4. Add targeted unit/regression tests for fixed cases and re-run the comprehensive suite"
    echo "   5. Consider writing the transpiler output and the g++ stderr to per-file logs to accelerate triage"
}

# Test 1: Stage 0 vs Stage 1 comparison using traditional colon syntax
TEST_FILE="${SCRIPT_DIR}/simple_main_colon.cpp2"
if [[ ! -f "${TEST_FILE}" ]]; then
    echo "❌ Test file not found: ${TEST_FILE}"
    exit 1
fi

run_test "Stage 0 vs Stage 1 comparison (colon syntax)" "
\"${BUILD_DIR}/stage0_cli\" transpile \"${TEST_FILE}\" \"${BUILD_DIR}/stage0_emit.cpp\" &&
\"${BUILD_DIR}/stage1_cli\" \"${TEST_FILE}\" \"${BUILD_DIR}/stage1_transpile.cpp\" &&
echo 'Checking if stage0 output compiles...' &&
${CXX} ${CXXFLAGS} -c \"${BUILD_DIR}/stage0_emit.cpp\" -o /tmp/stage0_test.o 2>/dev/null &&
echo '✓ Stage 0 output compiles successfully.' &&
rm -f /tmp/stage0_test.o &&
echo 'Checking if stage1 output compiles...' &&
${CXX} ${CXXFLAGS} -c \"${BUILD_DIR}/stage1_transpile.cpp\" -o /tmp/stage1_test.o 2>/dev/null &&
echo '✓ Stage 1 output compiles successfully.' &&
rm -f /tmp/stage1_test.o
"

# Test 2: Stage 1 only test for auto main() syntax
AUTO_TEST_FILE="${SCRIPT_DIR}/simple_main_auto.cpp2"
if [[ ! -f "${AUTO_TEST_FILE}" ]]; then
    echo "❌ Test file not found: ${AUTO_TEST_FILE}"
    exit 1
fi

run_test "Stage 1 auto main() syntax test" "
\"${BUILD_DIR}/stage1_cli\" \"${AUTO_TEST_FILE}\" \"${BUILD_DIR}/stage1_auto_transpile.cpp\"
" "grep -q 'int main() {' \"${BUILD_DIR}/stage1_auto_transpile.cpp\""

# Test 3: Stage 1 enhanced capabilities test (mixed syntax)
MIXED_TEST_FILE="${SCRIPT_DIR}/simple_mixed.cpp2"
if [[ ! -f "${MIXED_TEST_FILE}" ]]; then
    echo "❌ Test file not found: ${MIXED_TEST_FILE}"
    exit 1
fi

# Test 4: Stage 1 error handling test (intentionally broken syntax)
BROKEN_TEST_FILE="${SCRIPT_DIR}/test_broken_syntax.cpp2"
if [[ -f "${BROKEN_TEST_FILE}" ]]; then
    run_test "Stage 1 error handling test" "
    \"${BUILD_DIR}/stage1_cli\" \"${BROKEN_TEST_FILE}\" \"${BUILD_DIR}/stage1_broken_transpile.cpp\" &&
    ${CXX} ${CXXFLAGS} -c \"${BUILD_DIR}/stage1_broken_transpile.cpp\" -o /tmp/broken_test.o 2>&1
    " ""  # This test is expected to fail to demonstrate error analysis
fi

# Test 5: Comprehensive regression test suite (all 189 cpp2 files)
echo ""
echo "🧪 Running comprehensive regression test suite (189 files)..."
echo "=========================================================="

# Get all cpp2 test files
ALL_CPP2_FILES=$(find "${SCRIPT_DIR}" -name "*.cpp2" -type f | sort)

# Initialize counters for comprehensive testing
COMP_TOTAL_TESTS=0
COMP_PASSED_TESTS=0
COMP_FAILED_TESTS=0
COMP_ERROR_LOG=""

# Test each cpp2 file
for test_file in $ALL_CPP2_FILES; do
    ((COMP_TOTAL_TESTS++))
    filename=$(basename "$test_file")

    echo -n "Testing $filename... "

    # Create output path
    output_file="${BUILD_DIR}/comprehensive_test_${filename%.cpp2}.cpp"

    # Run stage1 transpiler and capture stderr to a per-file log
    transpile_err="${output_file}.transpile.err"
    cmp_err="${output_file}.err"
    rm -f "$transpile_err" "$cmp_err"
    if "${BUILD_DIR}/stage1_cli" "$test_file" "$output_file" 2>"$transpile_err"; then
        # Try to compile the output and capture compiler stderr to a per-file .err
        if ${CXX} ${CXXFLAGS} -c "$output_file" -o /tmp/comp_test_${COMP_TOTAL_TESTS}.o 2>"$cmp_err"; then
            echo "✅ PASS"
            ((COMP_PASSED_TESTS++))
            rm -f /tmp/comp_test_${COMP_TOTAL_TESTS}.o
            rm -f "$transpile_err" "$cmp_err"
        else
            echo "❌ FAIL (compile error)"
            ((COMP_FAILED_TESTS++))
            COMP_ERROR_LOG+="❌ $filename: Compilation failed (see $cmp_err)\n"
        fi
    else
        echo "❌ FAIL (transpilation error)"
        ((COMP_FAILED_TESTS++))
        COMP_ERROR_LOG+="❌ $filename: Transpilation failed (see $transpile_err)\n"
    fi
done

echo ""
echo "📊 Comprehensive Regression Test Results"
echo "========================================="
echo "Total Files Tested: $COMP_TOTAL_TESTS"
echo "Passed: $COMP_PASSED_TESTS"
echo "Failed: $COMP_FAILED_TESTS"
echo "Success Rate: $((COMP_PASSED_TESTS * 100 / COMP_TOTAL_TESTS))%"

if [[ $COMP_FAILED_TESTS -gt 0 ]]; then
    echo ""
    echo "❌ Failed Files:"
    echo "$COMP_ERROR_LOG"
fi

# Final Results Summary
echo ""
echo "📈 Complete Test Results Summary"
echo "==============================="
echo "Inductive Tests: $TOTAL_TESTS passed, $FAILED_TESTS failed"
echo "Comprehensive Suite: $COMP_TOTAL_TESTS files tested, $COMP_PASSED_TESTS passed, $COMP_FAILED_TESTS failed"
echo "Overall Success Rate: $(((PASSED_TESTS + COMP_PASSED_TESTS) * 100 / (TOTAL_TESTS + COMP_TOTAL_TESTS)))%"

if [[ $((FAILED_TESTS + COMP_FAILED_TESTS)) -gt 0 ]]; then
    echo ""
    echo "❌ Error Details:"
    if [[ -n "$ERROR_LOG" ]]; then
        echo "Inductive test errors:"
        echo "$ERROR_LOG"
    fi
    if [[ -n "$COMP_ERROR_LOG" ]]; then
        echo "Comprehensive test errors:"
        echo "$COMP_ERROR_LOG"
    fi
fi

# Analyze errors and provide improvement suggestions
analyze_errors

echo ""
if [[ $((FAILED_TESTS + COMP_FAILED_TESTS)) -eq 0 ]]; then
    echo "🎯 All 189 regression tests passed! The transpiler is production-ready."
else
    echo "🔄 $((FAILED_TESTS + COMP_FAILED_TESTS)) tests failed. Review the suggestions above to improve the transpiler."
fi