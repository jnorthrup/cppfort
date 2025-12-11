#!/usr/bin/env bash
# Run regression tests comparing cppfront and cppfort outputs

set -e

# Script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Paths
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
CPPFRONT_TESTS="${CPPFRONT_TESTS:-$HOME/src/cppfront/regression-tests}"
CPPFRONT_BIN="${CPPFRONT_BIN:-$(which cppfront)}"
CPPFORT_BIN="$BUILD_DIR/src/cppfort"
CORPUS_DIR="${CORPUS_DIR:-$PROJECT_ROOT/corpus}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Cppfort Regression Test Runner"
echo "=============================="
echo ""

# Check for cppfront
if [ ! -f "$CPPFRONT_BIN" ]; then
    echo -e "${RED}Error: cppfront binary not found at: $CPPFRONT_BIN${NC}"
    echo "Set CPPFRONT_BIN environment variable or install cppfront"
    exit 1
fi

# Check for cppfront tests
if [ ! -d "$CPPFRONT_TESTS" ]; then
    echo -e "${YELLOW}Warning: cppfront test directory not found at: $CPPFRONT_TESTS${NC}"
    echo "Cloning cppfront repository..."
    mkdir -p "$(dirname "$CPPFRONT_TESTS")"
    git clone https://github.com/hsutter/cppfront.git "$(dirname "$CPPFRONT_TESTS")/cppfront"
    CPPFRONT_TESTS="$(dirname "$CPPFRONT_TESTS")/cppfront/regression-tests"
fi

# Check for cppfort binary
if [ ! -f "$CPPFORT_BIN" ]; then
    echo -e "${RED}Error: cppfort binary not found at: $CPPFORT_BIN${NC}"
    echo "Build cppfort first: cmake --build build"
    exit 1
fi

# Create corpus directory
mkdir -p "$CORPUS_DIR"

# Run regression runner
echo "Running regression comparison..."
echo "  Cppfront tests: $CPPFRONT_TESTS"
echo "  Cppfront binary: $CPPFRONT_BIN"
echo "  Cppfort binary: $CPPFORT_BIN"
echo "  Corpus output: $CORPUS_DIR"
echo ""

"$BUILD_DIR/tests/regression_runner" \
    "$CPPFRONT_TESTS" \
    "$CPPFRONT_BIN" \
    "$CPPFORT_BIN" \
    "$CORPUS_DIR"

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo -e "${GREEN}Regression tests completed successfully${NC}"
    echo ""
    echo "Corpus generated at: $CORPUS_DIR"
    echo "  - cppfront/: Reference outputs from cppfront"
    echo "  - cppfort/: Outputs from cppfort"
    echo "  - diffs/: Isomorphic comparison results"
    echo "  - summary.csv: Statistical summary"
    echo "  - sha256_database.txt: File integrity checksums"
else
    echo -e "${RED}Regression tests failed with exit code $RESULT${NC}"
    exit $RESULT
fi

# Display summary if available
if [ -f "$CORPUS_DIR/summary.csv" ]; then
    echo ""
    echo "=== Summary Statistics ==="
    # Count lines in summary (excluding header)
    TOTAL=$(tail -n +2 "$CORPUS_DIR/summary.csv" | wc -l)
    BOTH_SUCCESS=$(tail -n +2 "$CORPUS_DIR/summary.csv" | awk -F, '$2=="1" && $3=="1"' | wc -l)
    EQUIVALENT=$(tail -n +2 "$CORPUS_DIR/summary.csv" | awk -F, '$4=="1"' | wc -l)

    echo "Total test cases: $TOTAL"
    echo "Both transpilers succeeded: $BOTH_SUCCESS"
    echo "Semantically equivalent: $EQUIVALENT"

    if [ "$BOTH_SUCCESS" -gt 0 ]; then
        EQUIV_PCT=$(awk "BEGIN {printf \"%.1f\", ($EQUIVALENT / $BOTH_SUCCESS) * 100}")
        echo "Equivalence rate: $EQUIV_PCT%"
    fi
fi