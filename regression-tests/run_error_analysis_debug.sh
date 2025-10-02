#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
BUILD_DIR="${PROJECT_ROOT}/build"
ANALYSIS_DIR="${BUILD_DIR}/error_analysis"

mkdir -p "${ANALYSIS_DIR}"

echo "Testing simple transpilation..."
echo "Using stage1_cli: ${BUILD_DIR}/stage1_cli"

# Test ONE file
CPP2_FILE="${SCRIPT_DIR}/simple_main_auto.cpp2"
CPP_OUT="/tmp/test_debug.cpp"

echo "Testing file: ${CPP2_FILE}"
echo "Output to: ${CPP_OUT}"

# Run with explicit timeout and capture output
if timeout 2 "${BUILD_DIR}/stage1_cli" "${CPP2_FILE}" "${CPP_OUT}" 2>&1; then
    echo "Transpilation succeeded"
    echo "Trying to compile..."
    if timeout 2 g++ -std=c++20 -c "${CPP_OUT}" -o "/tmp/test_debug.o" 2>&1; then
        echo "Compilation succeeded"
    else
        echo "Compilation failed"
    fi
else
    echo "Transpilation failed"
fi

echo "Done"