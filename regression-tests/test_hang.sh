#!/usr/bin/env bash
set -x

echo "START"
BUILD_DIR="/Users/jim/work/cppfort/build"
cpp2_file="/Users/jim/work/cppfort/regression-tests/simple_main_auto.cpp2"
cpp_output="/tmp/test_hang.cpp"

echo "Before stage1_cli"
timeout 2 "${BUILD_DIR}/stage1_cli" "${cpp2_file}" "${cpp_output}" </dev/null >/dev/null 2>&1
echo "After stage1_cli, exit code: $?"

echo "END"