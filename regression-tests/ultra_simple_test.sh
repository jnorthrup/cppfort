#!/usr/bin/env bash
set -x

SCRIPT_DIR="/Users/jim/work/cppfort/regression-tests"
BUILD_DIR="/Users/jim/work/cppfort/build"

TOTAL=0
for cpp2_file in "${SCRIPT_DIR}"/simple_main_auto.cpp2; do
    echo "Processing: $cpp2_file"
    ((TOTAL++))
    echo "TOTAL is now: $TOTAL"

    test_name=$(basename "${cpp2_file}" .cpp2)
    echo "Test name: $test_name"

    echo -n "  Testing ${test_name}... "

    # This is where it might hang
    "${BUILD_DIR}/stage1_cli" "${cpp2_file}" "/tmp/test.cpp" </dev/null >/dev/null 2>&1
    echo "Done with stage1_cli"
done

echo "Finished"