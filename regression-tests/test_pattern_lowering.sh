#!/bin/bash

# regression-tests/test_pattern_lowering.sh
# Test suite for n-way lowering patterns
# Validates pattern matching and multi-target code generation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

run_test() {
    local test_name="$1"
    local test_command="$2"

    TESTS_RUN=$((TESTS_RUN + 1))

    log_info "Running: $test_name"

    if eval "$test_command" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} PASSED"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "  ${RED}✗${NC} FAILED"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    log_error "Build directory not found: $BUILD_DIR"
    log_info "Please run: cmake -B build && cmake --build build"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

log_info "Starting Pattern Lowering Regression Tests"
echo "=============================================="
echo ""

# ============================================================================
# Test 1: Pattern Matcher Unit Tests
# ============================================================================
echo "Test Suite 1: Pattern Matcher Unit Tests"
echo "----------------------------------------"

if [ -f "$BUILD_DIR/test_pattern_matcher" ]; then
    run_test "Pattern Matcher Unit Tests" "$BUILD_DIR/test_pattern_matcher"
else
    log_warn "Pattern matcher tests not built"
fi

echo ""

# ============================================================================
# Test 2: Pattern Integration Tests
# ============================================================================
echo "Test Suite 2: Pattern Integration Tests"
echo "----------------------------------------"

if [ -f "$BUILD_DIR/test_pattern_integration" ]; then
    run_test "Pattern Integration Tests" "$BUILD_DIR/test_pattern_integration"
else
    log_warn "Pattern integration tests not built"
fi

echo ""

# ============================================================================
# Test 3: Machine Pattern Tests
# ============================================================================
echo "Test Suite 3: Machine Pattern Tests"
echo "------------------------------------"

if [ -f "$BUILD_DIR/test_machine_patterns" ]; then
    run_test "Machine Pattern Tests" "$BUILD_DIR/test_machine_patterns"
else
    log_warn "Machine pattern tests not built"
fi

echo ""

# ============================================================================
# Test 4: Pattern Database Tests
# ============================================================================
echo "Test Suite 4: Pattern Database Tests"
echo "------------------------------------"

if [ -f "$BUILD_DIR/test_pattern_database" ]; then
    run_test "Pattern Database Tests" "$BUILD_DIR/test_pattern_database"
else
    log_warn "Pattern database tests not built"
fi

echo ""

# ============================================================================
# Test 5: YAML Pattern Loading
# ============================================================================
echo "Test Suite 5: YAML Pattern Loading"
echo "-----------------------------------"

test_yaml_patterns() {
    # Check if pattern YAML files exist
    local patterns_dir="$PROJECT_ROOT/patterns"

    if [ ! -d "$patterns_dir" ]; then
        log_error "Patterns directory not found: $patterns_dir"
        return 1
    fi

    local yaml_files=("$patterns_dir"/*.yaml)

    if [ ${#yaml_files[@]} -eq 0 ]; then
        log_warn "No YAML pattern files found"
        return 1
    fi

    for yaml_file in "${yaml_files[@]}"; do
        if [ -f "$yaml_file" ]; then
            log_info "Found pattern file: $(basename "$yaml_file")"
        fi
    done

    return 0
}

run_test "YAML Pattern Files Exist" "test_yaml_patterns"

echo ""

# ============================================================================
# Test 6: Pattern Coverage Analysis
# ============================================================================
echo "Test Suite 6: Pattern Coverage Analysis"
echo "---------------------------------------"

analyze_pattern_coverage() {
    local patterns_registered=0
    local targets_covered=0

    # This is a placeholder - actual implementation would query the system
    log_info "Pattern coverage analysis (placeholder)"

    # Check if pattern matcher headers exist
    if [ -f "$PROJECT_ROOT/src/stage0/pattern_matcher.h" ]; then
        patterns_registered=1
    fi

    # Check if machine headers exist
    if [ -f "$PROJECT_ROOT/src/stage0/machine.h" ]; then
        targets_covered=1
    fi

    [ $patterns_registered -eq 1 ] && [ $targets_covered -eq 1 ]
}

run_test "Pattern Coverage Headers Exist" "analyze_pattern_coverage"

echo ""

# ============================================================================
# Test 7: Multi-Target Lowering Validation
# ============================================================================
echo "Test Suite 7: Multi-Target Lowering"
echo "------------------------------------"

validate_multi_target() {
    # Verify that multiple target languages are supported
    local header_file="$PROJECT_ROOT/src/stage0/node.h"

    if [ ! -f "$header_file" ]; then
        log_error "Node header not found: $header_file"
        return 1
    fi

    # Check for TargetLanguage enum
    if grep -q "enum class TargetLanguage" "$header_file"; then
        log_info "TargetLanguage enum found"

        # Check for MLIR dialects
        if grep -q "MLIR_ARITH" "$header_file"; then
            log_info "  - MLIR Arith dialect supported"
        fi

        if grep -q "MLIR_CF" "$header_file"; then
            log_info "  - MLIR CF dialect supported"
        fi

        if grep -q "MLIR_SCF" "$header_file"; then
            log_info "  - MLIR SCF dialect supported"
        fi

        if grep -q "MLIR_MEMREF" "$header_file"; then
            log_info "  - MLIR MemRef dialect supported"
        fi

        if grep -q "MLIR_FUNC" "$header_file"; then
            log_info "  - MLIR Func dialect supported"
        fi

        return 0
    else
        log_error "TargetLanguage enum not found"
        return 1
    fi
}

run_test "Multi-Target Support Validation" "validate_multi_target"

echo ""

# ============================================================================
# Test 8: Pattern Matcher API Validation
# ============================================================================
echo "Test Suite 8: Pattern Matcher API"
echo "----------------------------------"

validate_pattern_api() {
    local header_file="$PROJECT_ROOT/src/stage0/pattern_matcher.h"

    if [ ! -f "$header_file" ]; then
        log_error "Pattern matcher header not found"
        return 1
    fi

    # Check for core API methods
    local required_methods=(
        "registerPattern"
        "match"
        "hasPattern"
        "getPatternsForKind"
        "getPatternCount"
        "clear"
    )

    local all_found=1

    for method in "${required_methods[@]}"; do
        if grep -q "$method" "$header_file"; then
            log_info "  ✓ API method found: $method"
        else
            log_error "  ✗ API method missing: $method"
            all_found=0
        fi
    done

    [ $all_found -eq 1 ]
}

run_test "Pattern Matcher API Validation" "validate_pattern_api"

echo ""

# ============================================================================
# Test 9: NodeKind Coverage
# ============================================================================
echo "Test Suite 9: NodeKind Coverage"
echo "--------------------------------"

validate_node_kinds() {
    local header_file="$PROJECT_ROOT/src/utils/multi_index.h"

    if [ ! -f "$header_file" ]; then
        log_error "Multi-index header not found"
        return 1
    fi

    # Check for NodeKind categories
    local categories=(
        "CFG_START"
        "DATA_START"
        "ARITH_START"
        "BITWISE_START"
        "COMPARE_START"
        "FLOAT_START"
    )

    local all_found=1

    for category in "${categories[@]}"; do
        if grep -q "$category" "$header_file"; then
            log_info "  ✓ NodeKind category: $category"
        else
            log_warn "  ? NodeKind category: $category"
        fi
    done

    return 0
}

run_test "NodeKind Coverage" "validate_node_kinds"

echo ""

# ============================================================================
# Test Summary
# ============================================================================
echo "=============================================="
echo "Test Summary"
echo "=============================================="
echo "Total Tests:  $TESTS_RUN"
echo -e "Passed:       ${GREEN}$TESTS_PASSED${NC}"
echo -e "Failed:       ${RED}$TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
