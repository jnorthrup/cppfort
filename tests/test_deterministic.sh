#!/usr/bin/env bash
# Test deterministic compilation
# Verifies that identical sources produce identical outputs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."
BUILD_DIR="${PROJECT_ROOT}/build"
NWAY="${BUILD_DIR}/nway_compiler"

echo "═══════════════════════════════════════════════════════════"
echo "  Deterministic Compilation Test"
echo "═══════════════════════════════════════════════════════════"

# Test source
TEST_SOURCE="/tmp/deterministic_test.cpp2"
cat > "${TEST_SOURCE}" << 'EOF'
fibonacci: (n: int) -> int = {
    if n <= 1 {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}

main: () -> int = {
    return fibonacci(10);
}
EOF

# Compile twice with identical settings
echo ""
echo "Compiling source (first run)..."
"${NWAY}" "${TEST_SOURCE}" -o /tmp/test1.cpp --deterministic 2>&1 | grep -v "^$" || true

echo "Compiling source (second run)..."
"${NWAY}" "${TEST_SOURCE}" -o /tmp/test2.cpp --deterministic 2>&1 | grep -v "^$" || true

# Compare outputs
echo ""
echo "Comparing outputs..."
if diff -q /tmp/test1.cpp /tmp/test2.cpp >/dev/null; then
    echo "✓ PASS: Outputs are identical (deterministic)"
    RESULT=0
else
    echo "✗ FAIL: Outputs differ (non-deterministic)"
    echo ""
    echo "Differences:"
    diff /tmp/test1.cpp /tmp/test2.cpp || true
    RESULT=1
fi

# Test with different timestamps
echo ""
echo "Testing timestamp independence..."

export SOURCE_DATE_EPOCH=1000000000
"${NWAY}" "${TEST_SOURCE}" -o /tmp/test_t1.cpp --deterministic 2>&1 | grep -v "^$" || true

export SOURCE_DATE_EPOCH=2000000000
"${NWAY}" "${TEST_SOURCE}" -o /tmp/test_t2.cpp --deterministic 2>&1 | grep -v "^$" || true

if diff -q /tmp/test_t1.cpp /tmp/test_t2.cpp >/dev/null; then
    echo "✓ PASS: Timestamp-independent compilation"
else
    echo "✗ FAIL: Timestamp affects output"
    RESULT=1
fi

# Test hash reproducibility
echo ""
echo "Testing hash reproducibility..."

HASH1=$(sha256sum /tmp/test1.cpp | awk '{print $1}')
HASH2=$(sha256sum /tmp/test2.cpp | awk '{print $1}')

echo "Hash 1: ${HASH1}"
echo "Hash 2: ${HASH2}"

if [ "${HASH1}" = "${HASH2}" ]; then
    echo "✓ PASS: Hashes match (deterministic build)"
else
    echo "✗ FAIL: Hashes differ"
    RESULT=1
fi

# Clean up
rm -f /tmp/test1.cpp /tmp/test2.cpp /tmp/test_t1.cpp /tmp/test_t2.cpp "${TEST_SOURCE}"

echo ""
echo "═══════════════════════════════════════════════════════════"
if [ ${RESULT} -eq 0 ]; then
    echo "  ✓ ALL DETERMINISTIC TESTS PASSED"
else
    echo "  ✗ DETERMINISTIC TESTS FAILED"
fi
echo "═══════════════════════════════════════════════════════════"

exit ${RESULT}