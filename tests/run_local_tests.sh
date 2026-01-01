#!/bin/bash
# Local test runner for cppfort

set -e

CPPFORT="./build/src/cppfort"
TEST_DIR="tests/local"
PASSED=0
FAILED=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "=== CppFort Local Test Runner ==="
echo ""

for test_file in "$TEST_DIR"/*.cpp2; do
  if [ ! -f "$test_file" ]; then
    continue
  fi

  test_name=$(basename "$test_file" .cpp2)
  output_file="/tmp/${test_name}_output.cpp"

  echo -n "Testing $test_name ... "

  if timeout 1 "$CPPFORT" "$test_file" "$output_file" > /tmp/${test_name}_log.txt 2>&1; then
    echo -e "${GREEN}PASS${NC}"
    ((PASSED++))
  else
    echo -e "${RED}FAIL${NC}"
    echo "  Error output:"
    head -3 /tmp/${test_name}_log.txt | sed 's/^/    /'
    ((FAILED++))
  fi
done

echo ""
echo "=== Results ==="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "Total: $((PASSED + FAILED))"

if [ $FAILED -eq 0 ]; then
  echo -e "${GREEN}All tests passed!${NC}"
  exit 0
else
  echo -e "${RED}Some tests failed${NC}"
  exit 1
fi
