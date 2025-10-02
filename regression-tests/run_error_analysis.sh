#!/usr/bin/env bash
# ----------------------------------------------------------------------
# regression-tests/run_error_analysis.sh
# ----------------------------------------------------------------------
# Triple induction: Stage 1 → Stage 0 feedback loop
# Analyzes transpilation/compilation errors to guide Stage 0 improvements
# ----------------------------------------------------------------------

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

# Counters for different error categories
declare -A ERROR_CATEGORIES
ERROR_CATEGORIES["ast_generation"]=0
ERROR_CATEGORIES["type_conversion"]=0
ERROR_CATEGORIES["syntax_generation"]=0
ERROR_CATEGORIES["identifier_resolution"]=0
ERROR_CATEGORIES["template_instantiation"]=0
ERROR_CATEGORIES["unknown"]=0

declare -A CATEGORY_EXAMPLES

# Function to categorize compilation errors and map to Stage 0 components
categorize_error() {
    local error_msg="$1"
    local test_file="$2"

    case "$error_msg" in
        *"expected unqualified-id"*)
            ERROR_CATEGORIES["syntax_generation"]=$((${ERROR_CATEGORIES["syntax_generation"]} + 1))
            CATEGORY_EXAMPLES["syntax_generation"]="${test_file}: ${error_msg}"
            return 0
            ;;
        *"unknown type name"*|*"type specifier"*)
            ERROR_CATEGORIES["type_conversion"]=$((${ERROR_CATEGORIES["type_conversion"]} + 1))
            CATEGORY_EXAMPLES["type_conversion"]="${test_file}: ${error_msg}"
            return 0
            ;;
        *"use of undeclared identifier"*)
            ERROR_CATEGORIES["identifier_resolution"]=$((${ERROR_CATEGORIES["identifier_resolution"]} + 1))
            CATEGORY_EXAMPLES["identifier_resolution"]="${test_file}: ${error_msg}"
            return 0
            ;;
        *"no matching function"*|*"template"*)
            ERROR_CATEGORIES["template_instantiation"]=$((${ERROR_CATEGORIES["template_instantiation"]} + 1))
            CATEGORY_EXAMPLES["template_instantiation"]="${test_file}: ${error_msg}"
            return 0
            ;;
        *"expected expression"*|*"expected"*)
            ERROR_CATEGORIES["ast_generation"]=$((${ERROR_CATEGORIES["ast_generation"]} + 1))
            CATEGORY_EXAMPLES["ast_generation"]="${test_file}: ${error_msg}"
            return 0
            ;;
        *)
            ERROR_CATEGORIES["unknown"]=$((${ERROR_CATEGORIES["unknown"]} + 1))
            CATEGORY_EXAMPLES["unknown"]="${test_file}: ${error_msg}"
            return 0
            ;;
    esac
}

# Analyze a cpp2 file through Stage 1 transpilation and Stage 0 error mapping
analyze_file() {
    local cpp2_file="$1"
    local test_name=$(basename "${cpp2_file}" .cpp2)
    local cpp_output="${ANALYSIS_DIR}/${test_name}.cpp"
    local error_file="${ANALYSIS_DIR}/${test_name}.err"

    echo -n "  Testing ${test_name}... "

    # Stage 1: Try to transpile (with timeout)
    if ! timeout 5 "${BUILD_DIR}/stage1_cli" "${cpp2_file}" "${cpp_output}" >"${error_file}.transpile.out" 2>"${error_file}.transpile"; then
        # Transpilation failed - analyze transpiler errors
        if [[ -f "${error_file}.transpile" ]]; then
            while IFS= read -r line; do
                if [[ "$line" =~ error:|Error:|ERROR: ]]; then
                    categorize_error "$line" "${test_name}"
                fi
            done < "${error_file}.transpile"
        fi
        echo "FAILED (transpile)"
        return 1
    fi

    # Stage 1 succeeded, try compiling the output (with timeout)
    if ! timeout 5 g++ -std=c++20 -I"${PROJECT_ROOT}/include" -c "${cpp_output}" -o "${ANALYSIS_DIR}/${test_name}.o" 2>"${error_file}.compile"; then
        # Compilation failed - analyze C++ compiler errors
        if [[ -f "${error_file}.compile" ]]; then
            while IFS= read -r line; do
                if [[ "$line" =~ error: ]]; then
                    # Extract just the error message
                    error_only=$(echo "$line" | sed -E 's/^([^:]+):[0-9]+:[0-9]+: error: //g')
                    categorize_error "$error_only" "${test_name}"
                fi
            done < "${error_file}.compile"
        fi
        echo "FAILED (compile)"
        return 1
    fi

    echo "PASSED"
    return 0
}

# Test sample of regression files
echo ""
echo "Analyzing regression test files..."
TOTAL=0
PASSED=0

# Test simple files first
for test_file in "${SCRIPT_DIR}"/simple_*.cpp2 \
                 "${SCRIPT_DIR}"/mixed-hello.cpp2 \
                 "${SCRIPT_DIR}"/pure2-hello.cpp2 \
                 "${SCRIPT_DIR}"/pure2-stdio.cpp2; do
    if [[ -f "${test_file}" ]]; then
        ((TOTAL++))
        if analyze_file "${test_file}"; then
            ((PASSED++))
        fi
    fi
done

# Analyze ALL regression tests for complete feedback
echo ""
echo "Phase 2: Analyzing ALL regression tests..."
echo "──────────────────────────────────────────"

# Disable pipefail temporarily to collect all errors
set +e

# Count total files first
TOTAL_FILES=$(ls -1 "${SCRIPT_DIR}"/*.cpp2 2>/dev/null | wc -l)
echo "Found ${TOTAL_FILES} cpp2 files to analyze"

PHASE2_ANALYZED=0
for test_file in "${SCRIPT_DIR}"/*.cpp2; do
    if [[ -f "${test_file}" ]]; then
        # Skip files already analyzed in phase 1
        base_name=$(basename "${test_file}")
        if [[ "${base_name}" == simple_*.cpp2 ]] || \
           [[ "${base_name}" == "mixed-hello.cpp2" ]] || \
           [[ "${base_name}" == "pure2-hello.cpp2" ]] || \
           [[ "${base_name}" == "pure2-stdio.cpp2" ]]; then
            continue
        fi

        ((PHASE2_ANALYZED++))
        if analyze_file "${test_file}"; then
            ((PASSED++))
        fi
        ((TOTAL++))

        # Show progress every 10 files
        if [[ $((PHASE2_ANALYZED % 10)) -eq 0 ]]; then
            echo "  Progress: ${PHASE2_ANALYZED} files analyzed..."
        fi
    fi
done

echo "Phase 2 complete: Analyzed ${PHASE2_ANALYZED} additional files"
set -e

# Generate Stage 0 improvement recommendations
echo ""
echo "📊 Error Analysis Results (Stage 1→0 Feedback)"
echo "==============================================="
echo "Total Tests Analyzed: ${TOTAL}"
echo "Passed: ${PASSED}"
echo "Failed: $((TOTAL - PASSED))"
echo ""

echo "Error Categories (mapped to Stage 0 components):"
echo "─────────────────────────────────────────────────"

# Sort categories by frequency
for category in syntax_generation type_conversion ast_generation identifier_resolution template_instantiation unknown; do
    count=${ERROR_CATEGORIES[$category]}
    if [[ $count -gt 0 ]]; then
        echo ""
        echo "🔧 ${category}: ${count} errors"

        # Map to Stage 0 component
        case "$category" in
            "syntax_generation")
                echo "   → Stage 0 Component: emitter.cpp"
                echo "   → Action: Review fix_expression_tokens() and C++ code generation"
                ;;
            "type_conversion")
                echo "   → Stage 0 Component: parser.cpp, ast.h"
                echo "   → Action: Improve type parsing and representation in AST"
                ;;
            "ast_generation")
                echo "   → Stage 0 Component: parser.cpp"
                echo "   → Action: Enhance AST construction for complex expressions"
                ;;
            "identifier_resolution")
                echo "   → Stage 0 Component: parser.cpp, emitter.cpp"
                echo "   → Action: Improve scope tracking and name resolution"
                ;;
            "template_instantiation")
                echo "   → Stage 0 Component: emitter.cpp"
                echo "   → Action: Better template parameter handling in code generation"
                ;;
        esac

        # Show one example
        if [[ -n "${CATEGORY_EXAMPLES[$category]:-}" ]]; then
            echo "   Example: ${CATEGORY_EXAMPLES[$category]:0:80}..."
        fi
    fi
done

echo ""
echo "💡 Stage 0 Improvement Priority Queue:"
echo "────────────────────────────────────────"

# Create sorted list by error frequency
declare -a PRIORITIES
for category in "${!ERROR_CATEGORIES[@]}"; do
    count=${ERROR_CATEGORIES[$category]}
    if [[ $count -gt 0 ]]; then
        PRIORITIES+=("${count}:${category}")
    fi
done

IFS=$'\n' PRIORITIES=($(sort -rn <<<"${PRIORITIES[*]}"))
unset IFS

RANK=1
for priority in "${PRIORITIES[@]}"; do
    count=${priority%%:*}
    category=${priority#*:}
    echo "${RANK}. Fix ${category} (${count} occurrences)"
    ((RANK++))
done

echo ""
if [[ ${PASSED} -eq ${TOTAL} ]]; then
    echo "✅ All tests passed! Stage 0 handling all patterns correctly."
else
    echo "🔄 Stage 0 needs improvement. Use priority queue above to guide development."
fi