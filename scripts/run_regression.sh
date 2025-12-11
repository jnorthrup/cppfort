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

# Run regression runner (corpus builder + rule extractor)
echo "Running regression corpus builder..."
echo "  Cppfront tests: $CPPFRONT_TESTS"
echo "  Cppfront binary: $CPPFRONT_BIN"
echo "  Corpus output: $CORPUS_DIR"
echo ""

"$BUILD_DIR/tests/regression_runner" \
    "$CPPFRONT_TESTS" \
    "$CPPFRONT_BIN" \
    "$CORPUS_DIR"

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo -e "${GREEN}Corpus generation completed successfully${NC}"
    echo ""
    echo "Corpus generated at: $CORPUS_DIR"
    echo "  - outputs/: Cppfront transpiled outputs"
    echo "  - sha256_database.txt: Input/output integrity checksums"
    echo "  - transpile_rules.txt: Extracted transformation patterns"
    echo "  - report.txt: Summary report"
else
    echo -e "${RED}Corpus generation failed with exit code $RESULT${NC}"
    exit $RESULT
fi

# Display report if available
if [ -f "$CORPUS_DIR/report.txt" ]; then
    echo ""
    echo "=== Report ==="
    cat "$CORPUS_DIR/report.txt"
fi