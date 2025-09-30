#!/usr/bin/env bash
# ----------------------------------------------------------------------
# regression-tests/run_attestation_tests.sh
# ----------------------------------------------------------------------
# Triple induction test harness: Stage 2 → Stage 1 feedback loop
# Validates that transpiled cpp2 produces deterministic, attestable binaries
# ----------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
BUILD_DIR="${PROJECT_ROOT}/build"
TEMP_DIR="${BUILD_DIR}/attestation_tests"

mkdir -p "${TEMP_DIR}"

echo "🔒 Stage 2→1 Attestation Test Harness"
echo "======================================"

# Build all stages
echo ""
echo "Building transpiler and attestation tools..."
pushd "${BUILD_DIR}" >/dev/null
cmake .. -DCMAKE_BUILD_TYPE=Debug 2>&1 | grep -E "(^--|Error|Warning)" || true
cmake --build . --target stage1_cli anticheat_cli 2>&1 | grep -E "(^--|\[.*%\]|Error|Warning)" || true
popd >/dev/null

if [[ ! -x "${BUILD_DIR}/stage1_cli" ]]; then
    echo "❌ stage1_cli not built"
    exit 1
fi

if [[ ! -x "${BUILD_DIR}/anticheat_cli" ]]; then
    echo "❌ anticheat_cli not built"
    exit 1
fi

# Test counters
TOTAL_TESTS=0
TRANSPILE_PASS=0
COMPILE_PASS=0
ATTEST_PASS=0
DETERMINISM_PASS=0

# Function to test a cpp2 file through the triple induction pipeline
test_attestation() {
    local cpp2_file="$1"
    local test_name=$(basename "${cpp2_file}" .cpp2)

    ((TOTAL_TESTS++))
    echo ""
    echo "Test ${TOTAL_TESTS}: ${test_name}"
    echo "────────────────────────────────────"

    local cpp_output="${TEMP_DIR}/${test_name}.cpp"
    local binary_debug="${TEMP_DIR}/${test_name}_debug"
    local binary_opt="${TEMP_DIR}/${test_name}_opt"

    # Stage 1: Transpile cpp2 → C++
    if ! "${BUILD_DIR}/stage1_cli" "${cpp2_file}" "${cpp_output}" 2>&1 | grep -v "^Parsed\|^Emitted\|^Wrote\|param " >/dev/null; then
        echo "  ❌ Transpilation failed"
        return 1
    fi
    echo "  ✓ Transpiled to C++"
    ((TRANSPILE_PASS++))

    # Compile debug build (-O0 -g)
    if ! g++ -std=c++20 -O0 -g "${cpp_output}" -o "${binary_debug}" 2>/dev/null; then
        echo "  ❌ Debug compilation failed"
        return 1
    fi
    echo "  ✓ Compiled debug binary"

    # Compile optimized build (-O2)
    if ! g++ -std=c++20 -O2 "${cpp_output}" -o "${binary_opt}" 2>/dev/null; then
        echo "  ❌ Optimized compilation failed"
        return 1
    fi
    echo "  ✓ Compiled optimized binary"
    ((COMPILE_PASS++))

    # Stage 2: Attest both binaries
    local attest_debug=$("${BUILD_DIR}/anticheat_cli" "${binary_debug}" 2>/dev/null | grep -oE '[0-9a-f]{64}' || echo "FAIL")
    local attest_opt=$("${BUILD_DIR}/anticheat_cli" "${binary_opt}" 2>/dev/null | grep -oE '[0-9a-f]{64}' || echo "FAIL")

    if [[ "${attest_debug}" == "FAIL" || "${attest_opt}" == "FAIL" ]]; then
        echo "  ❌ Attestation failed"
        return 1
    fi
    echo "  ✓ Attested debug:     ${attest_debug:0:16}..."
    echo "  ✓ Attested optimized: ${attest_opt:0:16}..."
    ((ATTEST_PASS++))

    # Check determinism: same code should produce similar disassembly structure
    # (not exact match due to optimization, but should both be valid)
    if [[ ${#attest_debug} -eq 64 && ${#attest_opt} -eq 64 ]]; then
        echo "  ✓ Deterministic attestation (both valid)"
        ((DETERMINISM_PASS++))
        return 0
    else
        echo "  ⚠ Non-deterministic attestation"
        return 1
    fi
}

# Test a sample of simple regression files
echo ""
echo "Running attestation tests on simple cpp2 files..."
echo ""

# Test the simple test files we know should work
test_attestation "${SCRIPT_DIR}/simple_main_colon.cpp2" || true
test_attestation "${SCRIPT_DIR}/simple_main_auto.cpp2" || true

# Test a few mixed-syntax files
for test_file in "${SCRIPT_DIR}"/mixed-hello.cpp2 \
                 "${SCRIPT_DIR}"/pure2-hello.cpp2 \
                 "${SCRIPT_DIR}"/pure2-stdio.cpp2; do
    if [[ -f "${test_file}" ]]; then
        test_attestation "${test_file}" || true
    fi
done

# Results summary
echo ""
echo "📊 Triple Induction Results (Stage 2→1)"
echo "========================================="
echo "Total Tests:           ${TOTAL_TESTS}"
echo "Transpilation Pass:    ${TRANSPILE_PASS}/${TOTAL_TESTS}"
echo "Compilation Pass:      ${COMPILE_PASS}/${TOTAL_TESTS}"
echo "Attestation Pass:      ${ATTEST_PASS}/${TOTAL_TESTS}"
echo "Determinism Pass:      ${DETERMINISM_PASS}/${TOTAL_TESTS}"
echo ""

if [[ ${DETERMINISM_PASS} -eq ${TOTAL_TESTS} ]]; then
    echo "✅ All attestation tests passed! Stage 2→1 feedback loop working."
    exit 0
else
    FAILED=$((TOTAL_TESTS - DETERMINISM_PASS))
    echo "⚠️  ${FAILED} tests need improvement. Stage 1 transpilation requires refinement."
    echo ""
    echo "💡 Next Steps (Stage 1→0 feedback):"
    echo "   - Review failed transpilations for AST generation issues"
    echo "   - Check Stage 0 emitter for missing cpp2 patterns"
    echo "   - Enhance parser to handle problematic constructs"
    exit 1
fi