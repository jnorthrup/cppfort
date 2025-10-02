#!/usr/bin/env bash
# Minimal working error analysis

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
BUILD_DIR="${PROJECT_ROOT}/build"
STAGE1="${BUILD_DIR}/stage1_cli"

echo "Error Analysis - Simple Version"
echo "================================"

if [[ ! -x "${STAGE1}" ]]; then
    echo "ERROR: stage1_cli not found at ${STAGE1}"
    exit 1
fi

TOTAL=0
PASSED=0

# Test each cpp2 file
for cpp2_file in "${SCRIPT_DIR}"/*.cpp2; do
    [[ ! -f "${cpp2_file}" ]] && continue

    name=$(basename "${cpp2_file}" .cpp2)
    ((TOTAL++))

    printf "[%3d] %-50s " "${TOTAL}" "${name}"

    # Transpile
    if timeout 2 "${STAGE1}" "${cpp2_file}" "/tmp/${name}.cpp" >/dev/null 2>&1; then
        # Compile
        if timeout 2 g++ -std=c++20 -I"${PROJECT_ROOT}/include" \
            -c "/tmp/${name}.cpp" -o "/tmp/${name}.o" 2>&1 | grep -q error; then
            echo "FAIL (compile)"
        else
            echo "PASS"
            ((PASSED++))
        fi
    else
        echo "FAIL (transpile)"
    fi
done

echo ""
echo "Results: ${PASSED}/${TOTAL} passed ($((PASSED * 100 / TOTAL))%)"