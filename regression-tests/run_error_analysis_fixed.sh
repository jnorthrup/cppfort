#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
BUILD_DIR="${PROJECT_ROOT}/build"
ANALYSIS_DIR="${BUILD_DIR}/error_analysis"

mkdir -p "${ANALYSIS_DIR}"

echo "🔍 Stage 1→0 Error Analysis Framework"
echo "======================================"

# Build stages
echo ""
echo "Building stage1_cli..."
pushd "${BUILD_DIR}" >/dev/null
cmake .. -DCMAKE_BUILD_TYPE=Debug 2>&1 | grep -E "(^--|Error|Warning)" || true
cmake --build . --target stage1_cli 2>&1 | grep -E "(^--|\[.*%\]|Error|Warning)" || true
popd >/dev/null

if [[ ! -x "${BUILD_DIR}/stage1_cli" ]]; then
    echo "❌ stage1_cli not built"
    exit 1
fi

# Counters
AST_ERRORS=0
TYPE_ERRORS=0
SYNTAX_ERRORS=0
ID_ERRORS=0
TEMPLATE_ERRORS=0
UNKNOWN_ERRORS=0

echo ""
echo "Analyzing regression test files..."
TOTAL=0
PASSED=0

# Process each test file
for cpp2_file in "${SCRIPT_DIR}"/simple_*.cpp2 "${SCRIPT_DIR}"/mixed-hello.cpp2 "${SCRIPT_DIR}"/pure2-hello.cpp2; do
    if [[ ! -f "${cpp2_file}" ]]; then
        continue
    fi

    ((TOTAL++))
    test_name=$(basename "${cpp2_file}" .cpp2)
    cpp_output="${ANALYSIS_DIR}/${test_name}.cpp"

    echo -n "  Testing ${test_name}... "

    # Try to transpile with timeout (redirect stdin to avoid hang)
    if timeout 5 "${BUILD_DIR}/stage1_cli" "${cpp2_file}" "${cpp_output}" </dev/null >/dev/null 2>&1; then
        # Try to compile
        if timeout 5 g++ -std=c++20 -I"${PROJECT_ROOT}/include" -c "${cpp_output}" -o "${ANALYSIS_DIR}/${test_name}.o" 2>"${ANALYSIS_DIR}/${test_name}.err"; then
            echo "PASSED"
            ((PASSED++))
        else
            echo "FAILED (compile)"
            # Analyze errors
            while IFS= read -r line; do
                if [[ "$line" =~ error: ]]; then
                    case "$line" in
                        *"expected unqualified-id"*) ((SYNTAX_ERRORS++)) ;;
                        *"unknown type name"*|*"type specifier"*) ((TYPE_ERRORS++)) ;;
                        *"use of undeclared identifier"*) ((ID_ERRORS++)) ;;
                        *"no matching function"*|*"template"*) ((TEMPLATE_ERRORS++)) ;;
                        *"expected expression"*|*"expected"*) ((AST_ERRORS++)) ;;
                        *) ((UNKNOWN_ERRORS++)) ;;
                    esac
                fi
            done < "${ANALYSIS_DIR}/${test_name}.err"
        fi
    else
        echo "FAILED (transpile)"
    fi
done

echo ""
echo "📊 Error Analysis Results (Stage 1→0 Feedback)"
echo "==============================================="
echo "Total Tests: ${TOTAL}"
echo "Passed: ${PASSED}"
echo "Failed: $((TOTAL - PASSED))"
echo ""

if [[ $((AST_ERRORS + TYPE_ERRORS + SYNTAX_ERRORS + ID_ERRORS + TEMPLATE_ERRORS + UNKNOWN_ERRORS)) -gt 0 ]]; then
    echo "Error Categories:"
    echo "─────────────────"
    [[ ${SYNTAX_ERRORS} -gt 0 ]] && echo "  Syntax Generation: ${SYNTAX_ERRORS}"
    [[ ${TYPE_ERRORS} -gt 0 ]] && echo "  Type Conversion: ${TYPE_ERRORS}"
    [[ ${AST_ERRORS} -gt 0 ]] && echo "  AST Generation: ${AST_ERRORS}"
    [[ ${ID_ERRORS} -gt 0 ]] && echo "  Identifier Resolution: ${ID_ERRORS}"
    [[ ${TEMPLATE_ERRORS} -gt 0 ]] && echo "  Template Instantiation: ${TEMPLATE_ERRORS}"
    [[ ${UNKNOWN_ERRORS} -gt 0 ]] && echo "  Unknown: ${UNKNOWN_ERRORS}"
fi

echo ""
if [[ ${PASSED} -eq ${TOTAL} ]]; then
    echo "✅ All tests passed!"
else
    echo "🔧 ${PASSED}/${TOTAL} tests passing. Fix Stage 0 transpiler for better results."
fi