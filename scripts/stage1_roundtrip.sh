#!/usr/bin/env bash
# scripts/stage1_roundtrip.sh
# Usage: scripts/stage1_roundtrip.sh /path/to/stage1_binary [outdir]
set -euo pipefail

STAGE1_BIN="${1:?path to stage1 binary required}"
OUTDIR="${2:-${PWD}/build/stage1_out}"

mkdir -p "$OUTDIR"
rm -rf "$OUTDIR"/*
"$STAGE1_BIN" --emit-dir "$OUTDIR"

cat > "$OUTDIR/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.10)
project(stage1_roundtrip LANGUAGES CXX)
file(GLOB SRCS "*.cpp" "*.cc" "*.cxx")
add_executable(stage1_roundtrip ${SRCS})
set_target_properties(stage1_roundtrip PROPERTIES CXX_STANDARD 17)
enable_testing()
add_test(NAME stage1_run COMMAND stage1_roundtrip)
CMAKE

pushd "$OUTDIR" >/dev/null
mkdir -p build
cmake -S . -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build -- -j$(sysctl -n hw.ncpu || echo 4)
ctest --test-dir build --output-on-failure
popd >/dev/null

echo "Round-trip build + test passed for emitted sources in: $OUTDIR"