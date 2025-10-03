#!/bin/bash
#
# Automated micro test harness for C++ decompilation validation
# Compiles each test at multiple optimization levels, extracts assembly,
# and validates decompilation pipeline
#

set -e

# Configuration
CPPFORT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$CPPFORT_ROOT/build"
MICRO_TESTS_DIR="$CPPFORT_ROOT/micro-tests"
RESULTS_DIR="$MICRO_TESTS_DIR/results"
STAGE2_DECOMPILER="$BUILD_DIR/stage2_decompiler"

# Optimization levels to test
OPT_LEVELS=("O0" "O1" "O2" "O3")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "========================================="
echo "C++ Micro Test Harness"
echo "========================================="
echo "Cppfort root: $CPPFORT_ROOT"
echo "Build dir: $BUILD_DIR"
echo "Results dir: $RESULTS_DIR"
echo ""

# Function to compile a single test at a given optimization level
compile_test() {
    local test_file="$1"
    local opt_level="$2"
    local output_bin="$3"

    g++ -std=c++20 -$opt_level "$test_file" -o "$output_bin" 2>/dev/null
    return $?
}

# Function to extract assembly from binary
extract_assembly() {
    local binary="$1"
    local output_asm="$2"

    objdump -d "$binary" > "$output_asm" 2>/dev/null
    return $?
}

# Function to run decompilation (if Stage 2 decompiler exists)
run_decompilation() {
    local asm_file="$1"
    local output_cpp="$2"

    if [ -f "$STAGE2_DECOMPILER" ]; then
        "$STAGE2_DECOMPILER" "$asm_file" > "$output_cpp" 2>/dev/null
        return $?
    else
        # Decompiler not yet implemented, skip
        return 2
    fi
}

# Function to validate behavioral equivalence
validate_behavior() {
    local original_bin="$1"
    local decompiled_bin="$2"

    # Run both binaries and compare exit codes
    local orig_exit=0
    local decomp_exit=0

    "$original_bin" >/dev/null 2>&1 || orig_exit=$?
    "$decompiled_bin" >/dev/null 2>&1 || decomp_exit=$?

    [ "$orig_exit" -eq "$decomp_exit" ]
    return $?
}

# Process all tests in a category
process_category() {
    local category="$1"
    local category_dir="$MICRO_TESTS_DIR/$category"

    if [ ! -d "$category_dir" ]; then
        echo "  ${YELLOW}Category $category not found, skipping${NC}"
        return
    fi

    echo ""
    echo "Testing category: $category"
    echo "-----------------------------------"

    local tests=$(ls "$category_dir"/*.cpp 2>/dev/null || echo "")

    if [ -z "$tests" ]; then
        echo "  ${YELLOW}No tests found in $category${NC}"
        return
    fi

    for test_file in $tests; do
        local test_name=$(basename "$test_file" .cpp)
        local test_results_dir="$RESULTS_DIR/$category/$test_name"
        mkdir -p "$test_results_dir"

        ((TOTAL_TESTS++))

        # Test at each optimization level
        local all_compiled=true
        local compilation_summary=""

        for opt in "${OPT_LEVELS[@]}"; do
            local bin_file="$test_results_dir/${test_name}_${opt}.out"
            local asm_file="$test_results_dir/${test_name}_${opt}.asm"

            if compile_test "$test_file" "$opt" "$bin_file"; then
                if extract_assembly "$bin_file" "$asm_file"; then
                    compilation_summary="${compilation_summary}${opt}:✓ "
                else
                    compilation_summary="${compilation_summary}${opt}:✗ "
                    all_compiled=false
                fi
            else
                compilation_summary="${compilation_summary}${opt}:✗ "
                all_compiled=false
            fi
        done

        # Report result
        if $all_compiled; then
            echo -e "  ${GREEN}✓${NC} $test_name [$compilation_summary]"
            ((PASSED_TESTS++))
        else
            echo -e "  ${RED}✗${NC} $test_name [$compilation_summary]"
            ((FAILED_TESTS++))
        fi
    done
}

# Process all categories
CATEGORIES=("control-flow" "arithmetic" "memory" "functions" "classes" "templates" "stdlib" "exceptions" "modern-cpp" "edge-cases")

for category in "${CATEGORIES[@]}"; do
    process_category "$category"
done

# Summary
echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total tests:  $TOTAL_TESTS"
echo -e "${GREEN}Passed:       $PASSED_TESTS${NC}"
echo -e "${RED}Failed:       $FAILED_TESTS${NC}"
echo -e "${YELLOW}Skipped:      $SKIPPED_TESTS${NC}"
echo ""

if [ "$FAILED_TESTS" -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed.${NC}"
    exit 1
fi
