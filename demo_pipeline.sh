#!/usr/bin/env bash
# ------------------------------------------------------------
# demo_pipeline.sh – Quick demonstration of the CPP2 → C++ pipeline
# ------------------------------------------------------------
#
# Prerequisites:
#   - Project built with CMake (provides `stage0` and `stage1` executables)
#   - A .cpp2 source file (e.g., `example.cpp2`)
#
# Usage:
#   ./demo_pipeline.sh example.cpp2
#
# The script performs:
#   1. Stage 0: transpile .cpp2 → .cpp (C++)
#   2. Compile the Stage 0 output with g++
#   3. Stage 1: run the stub transpiler on the same .cpp2 file
#   4. Compile the Stage 1 output with g++
#   5. Execute both binaries for a quick sanity check
# ------------------------------------------------------------

set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 <source.cpp2>"
    exit 1
fi

SRC_CPP2="$1"
BASE_NAME="$(basename "${SRC_CPP2}" .cpp2)"
BUILD_DIR="$(mktemp -d)"

# -----------------------------------------------------------------
# Stage 0 – Full transpilation
# -----------------------------------------------------------------
STAGE0_OUT="${BUILD_DIR}/${BASE_NAME}_stage0.cpp"
STAGE0_EXE="${BUILD_DIR}/${BASE_NAME}_stage0_exe"

echo "[Stage0] Transpiling ${SRC_CPP2} → ${STAGE0_OUT}"
./build/stage0_cli transpile "${SRC_CPP2}" "${STAGE0_OUT}"

echo "[Stage0] Compiling ${STAGE0_OUT} → ${STAGE0_EXE}"
g++ -std=c++20 -O0 -g "${STAGE0_OUT}" -o "${STAGE0_EXE}"

echo "[Stage0] Executing ${STAGE0_EXE}"
"${STAGE0_EXE}"

# -----------------------------------------------------------------
# Stage 1 – Stub transpilation (currently produces a simple stub)
# -----------------------------------------------------------------
STAGE1_OUT="${BUILD_DIR}/${BASE_NAME}_stage1.cpp"
STAGE1_EXE="${BUILD_DIR}/${BASE_NAME}_stage1_exe"

echo "[Stage1] Running stub transpiler on ${SRC_CPP2} → ${STAGE1_OUT}"
./build/stage1_cli "${SRC_CPP2}" "${STAGE1_OUT}"

echo "[Stage1] Compiling ${STAGE1_OUT} → ${STAGE1_EXE}"
g++ -std=c++20 -O0 -g "${STAGE1_OUT}" -o "${STAGE1_EXE}"

echo "[Stage1] Executing ${STAGE1_EXE}"
"${STAGE1_EXE}"

echo "Demo completed successfully."