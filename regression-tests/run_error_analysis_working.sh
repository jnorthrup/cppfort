#!/usr/bin/env bash
# Fixed error analysis script with proper error handling

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
BUILD_DIR="${PROJECT_ROOT}/build"
ANALYSIS_DIR="${BUILD_DIR}/error_analysis"

mkdir -p "${ANALYSIS_DIR}"

echo "🔍 Stage 1→0 Error Analysis Framework (FIXED)"
echo "=============================================="

# Verify stage1_cli exists
if [[ ! -x "${BUILD_DIR}/stage1_cli" ]]; then
    echo "❌ stage1_cli not found at ${BUILD_DIR}/stage1_cli"
    echo "Building it now..."
    pushd "${BUILD_DIR}" >/dev/null
    cmake .. -DCMAKE_BUILD_TYPE=Debug
    cmake --build . --target stage1_cli
    popd >/dev/null
fi

if [[ ! -x "${BUILD_DIR}/stage1_cli" ]]; then
    echo "❌ Failed to build stage1_cli"
    exit 1
fi

echo "✓ Using stage1_cli: ${BUILD_DIR}/stage1_cli"

# Error counters
declare -A ERROR_CATEGORIES=(
    ["ast_generation"]=0
    ["type_conversion"]=0
    ["syntax_generation"]=0
    ["identifier_resolution"]=0
    ["template_instantiation"]=0
    ["unknown"]=0
)

declare -A CATEGORY_EXAMPLES

# Categorize error function
categorize_error() {
    local error_msg="$1"
    local test_file="$2"

    case "$error_msg" in
        *"expected unqualified-id"*)
            ((ERROR_CATEGORIES["syntax_generation"]++))
            CATEGORY_EXAMPLES["syntax_generation"]="${test_file}: ${error_msg}"
            ;;
        *"unknown type name"*|*"type specifier"*)
            ((ERROR_CATEGORIES["type_conversion"]++))
            CATEGORY_EXAMPLES["type_conversion"]="${test_file}: ${error_msg}"
            ;;
        *"use of undeclared identifier"*)
            ((ERROR_CATEGORIES["identifier_resolution"]++))
            CATEGORY_EXAMPLES["identifier_resolution"]="${test_file}: ${error_msg}"
            ;;
        *"no matching function"*|*"template"*)
            ((ERROR_CATEGORIES["template_instantiation"]++))
            CATEGORY_EXAMPLES["template_instantiation"]="${test_file}: ${error_msg}"
            ;;
        *"expected expression"*|*"expected"*)
            ((ERROR_CATEGORIES["ast_generation"]++))
            CATEGORY_EXAMPLES["ast_generation"]="${test_file}: ${error_msg}"
            ;;
        *)
            ((ERROR_CATEGORIES["unknown"]++))
            CATEGORY_EXAMPLES["unknown"]="${test_file}: ${error_msg}"
            ;;
    esac
}

# Main analysis
echo ""
echo "Analyzing cpp2 files..."
echo "──────────────────────"

TOTAL=0
PASSED=0
FAILED_TRANSPILE=0
FAILED_COMPILE=0

# Process all cpp2 files
for cpp2_file in "${SCRIPT_DIR}"/*.cpp2; do
    if [[ ! -f "${cpp2_file}" ]]; then
        continue
    fi

    test_name=$(basename "${cpp2_file}" .cpp2)
    cpp_output="${ANALYSIS_DIR}/${test_name}.cpp"

    ((TOTAL++))
    echo -n "  [${TOTAL}] ${test_name}... "

    # Try transpilation with timeout
    if timeout 3 "${BUILD_DIR}/stage1_cli" "${cpp2_file}" "${cpp_output}" \
        >"${ANALYSIS_DIR}/${test_name}.transpile.out" \
        2>"${ANALYSIS_DIR}/${test_name}.transpile.err"; then

        # Transpilation succeeded, try compilation
        if timeout 3 g++ -std=c++20 -I"${PROJECT_ROOT}/include" \
            -c "${cpp_output}" -o "${ANALYSIS_DIR}/${test_name}.o" \
            2>"${ANALYSIS_DIR}/${test_name}.compile.err"; then

            echo "PASSED"
            ((PASSED++))
        else
            echo "FAILED (compile)"
            ((FAILED_COMPILE++))

            # Analyze compilation errors
            while IFS= read -r line; do
                if [[ "$line" =~ error: ]]; then
                    error_only=$(echo "$line" | sed -E 's/^[^:]+:[0-9]+:[0-9]+: error: //g')
                    categorize_error "$error_only" "${test_name}"
                fi
            done < "${ANALYSIS_DIR}/${test_name}.compile.err"
        fi
    else
        echo "FAILED (transpile)"
        ((FAILED_TRANSPILE++))

        # Analyze transpiler errors
        if [[ -s "${ANALYSIS_DIR}/${test_name}.transpile.err" ]]; then
            while IFS= read -r line; do
                if [[ "$line" =~ [Ee]rror ]]; then
                    categorize_error "$line" "${test_name}"
                fi
            done < "${ANALYSIS_DIR}/${test_name}.transpile.err"
        fi
    fi

    # Show progress for large test sets
    if [[ $((TOTAL % 20)) -eq 0 ]]; then
        echo "    Progress: ${TOTAL} files processed..."
    fi
done

# Results
echo ""
echo "📊 Error Analysis Results"
echo "========================="
echo "Total Tests: ${TOTAL}"
echo "✅ Passed: ${PASSED} ($((PASSED * 100 / TOTAL))%)"
echo "❌ Failed (transpile): ${FAILED_TRANSPILE}"
echo "❌ Failed (compile): ${FAILED_COMPILE}"
echo ""

echo "Error Categories:"
echo "─────────────────"
for category in syntax_generation type_conversion ast_generation identifier_resolution template_instantiation unknown; do
    count=${ERROR_CATEGORIES[$category]}
    if [[ $count -gt 0 ]]; then
        echo "• ${category}: ${count} errors"

        # Map to Stage 0 component
        case "$category" in
            "syntax_generation")
                echo "  → Fix in: emitter.cpp (code generation)"
                ;;
            "type_conversion")
                echo "  → Fix in: parser.cpp, ast.h (type handling)"
                ;;
            "ast_generation")
                echo "  → Fix in: parser.cpp (AST construction)"
                ;;
            "identifier_resolution")
                echo "  → Fix in: parser.cpp, emitter.cpp (scope/names)"
                ;;
            "template_instantiation")
                echo "  → Fix in: emitter.cpp (template handling)"
                ;;
        esac

        if [[ -n "${CATEGORY_EXAMPLES[$category]:-}" ]]; then
            echo "  Example: ${CATEGORY_EXAMPLES[$category]:0:70}..."
        fi
    fi
done

echo ""
echo "💡 Priority Queue (by frequency):"
echo "─────────────────────────────────"

# Create sorted priority list
declare -a PRIORITIES
for category in "${!ERROR_CATEGORIES[@]}"; do
    count=${ERROR_CATEGORIES[$category]}
    if [[ $count -gt 0 ]]; then
        PRIORITIES+=("$(printf "%05d:%s" "$count" "$category")")
    fi
done

if [[ ${#PRIORITIES[@]} -gt 0 ]]; then
    IFS=$'\n' SORTED=($(sort -rn <<<"${PRIORITIES[*]}"))
    unset IFS

    RANK=1
    for priority in "${SORTED[@]}"; do
        count=${priority%%:*}
        count=$((10#$count))  # Remove leading zeros
        category=${priority#*:}
        echo "${RANK}. Fix ${category} (${count} occurrences)"
        ((RANK++))
    done
else
    echo "No errors found - all tests passing!"
fi

echo ""
echo "Analysis complete. Error logs saved in: ${ANALYSIS_DIR}"