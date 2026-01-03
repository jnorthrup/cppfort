#!/bin/bash
# Build and run combinator test suite

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_DIR="$PROJECT_ROOT/tests"
BUILD_DIR="$PROJECT_ROOT/build"
CPPFORT="$BUILD_DIR/src/cppfort"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Compiler flags
CXX=${CXX:-clang++}
CXXFLAGS="-std=c++20 -O2 -I$PROJECT_ROOT/include"
ASAN_FLAGS="-fsanitize=address -fno-omit-frame-pointer"

echo "=== Combinator Test Suite ==="
echo ""

# List of cpp2 test files
CPP2_TESTS=(
    "structural_combinators_test"
    "transformation_combinators_test" 
    "reduction_combinators_test"
    "parsing_combinators_test"
    "pipeline_operator_test"
    "bytebuffer_test"
    "strview_test"
    "lazy_iterator_test"
    "combinator_laws_test"
    "benchmark_combinators"
    "zero_copy_verification_test"
    "combinator_integration_test"
)

PASSED=0
FAILED=0
SKIPPED=0

# Step 1: Transpile cpp2 files to cpp
echo "--- Transpiling cpp2 tests ---"
for test in "${CPP2_TESTS[@]}"; do
    cpp2_file="$TEST_DIR/${test}.cpp2"
    cpp_file="$TEST_DIR/${test}.cpp"
    
    if [ ! -f "$cpp2_file" ]; then
        echo -e "${YELLOW}SKIP${NC} $test.cpp2 (not found)"
        ((SKIPPED++))
        continue
    fi
    
    echo -n "Transpiling $test.cpp2 ... "
    if "$CPPFORT" "$cpp2_file" "$cpp_file" > /tmp/transpile_${test}.log 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAIL${NC}"
        cat /tmp/transpile_${test}.log | head -5
        ((FAILED++))
        continue
    fi
done

echo ""
echo "--- Compiling and running tests ---"

# Step 2: Compile and run each test
for test in "${CPP2_TESTS[@]}"; do
    cpp_file="$TEST_DIR/${test}.cpp"
    exe_file="$TEST_DIR/${test}"
    
    if [ ! -f "$cpp_file" ]; then
        continue
    fi
    
    echo -n "Building $test ... "
    
    # Compile with AddressSanitizer for verification tests
    if [[ "$test" == *"zero_copy"* || "$test" == *"verification"* ]]; then
        EXTRA_FLAGS="$ASAN_FLAGS"
    else
        EXTRA_FLAGS=""
    fi
    
    if $CXX $CXXFLAGS $EXTRA_FLAGS -o "$exe_file" "$cpp_file" > /tmp/compile_${test}.log 2>&1; then
        echo -e "${GREEN}OK${NC}"
        
        # Run the test
        echo -n "Running $test ... "
        if timeout 30 "$exe_file" > /tmp/run_${test}.log 2>&1; then
            echo -e "${GREEN}PASS${NC}"
            ((PASSED++))
        else
            echo -e "${RED}FAIL${NC}"
            tail -10 /tmp/run_${test}.log | sed 's/^/  /'
            ((FAILED++))
        fi
    else
        echo -e "${RED}FAIL${NC}"
        tail -10 /tmp/compile_${test}.log | sed 's/^/  /'
        ((FAILED++))
    fi
done

echo ""
echo "=== Results ==="
echo -e "Passed:  ${GREEN}$PASSED${NC}"
echo -e "Failed:  ${RED}$FAILED${NC}"
echo -e "Skipped: ${YELLOW}$SKIPPED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All combinator tests passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED tests failed${NC}"
    exit 1
fi
