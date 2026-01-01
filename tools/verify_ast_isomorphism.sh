#!/bin/bash
# AST Isomorphism Verification Script
# Compares cppfort output AST against reference AST from cppfront

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPPFORT="$SCRIPT_DIR/../build/src/cppfort"
CPPFRONT="$SCRIPT_DIR/../third_party/cppfront/source/cppfront"
CORPUS_DIR="$SCRIPT_DIR/../corpus"
CLANG++="clang++"

usage() {
    echo "Usage: $0 <input.cpp2> [--verbose]"
    echo ""
    echo "Verifies cppfort output against cppfront using AST isomorphism"
    echo ""
    echo "Arguments:"
    echo "  input.cpp2    Cpp2 source file to test"
    echo "  --verbose     Show detailed AST comparison"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

INPUT_FILE="$1"
VERBOSE=0

if [ "$2" = "--verbose" ]; then
    VERBOSE=1
fi

# Validate input
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Get base name
BASENAME=$(basename "$INPUT_FILE" .cpp2)
REF_DIR="$CORPUS_DIR/reference"
REF_AST_DIR="$CORPUS_DIR/reference_ast"
ISOMORPH_DIR="$CORPUS_DIR/isomorphs"

# Check if reference files exist
REF_CPP="$REF_DIR/${BASENAME}.cpp"
REF_AST="$REF_AST_DIR/${BASENAME}.ast.txt"
REF_ISOMORPH="$ISOMORPH_DIR/${BASENAME}.isomorph.json"

if [ ! -f "$REF_CPP" ]; then
    echo "Warning: Reference C++ not found: $REF_CPP"
    echo "Will generate from cppfront..."
fi

if [ ! -f "$REF_AST" ]; then
    echo "Error: Reference AST not found: $REF_AST"
    echo "Cannot perform AST verification"
    exit 1
fi

if [ ! -f "$REF_ISOMORPH" ]; then
    echo "Warning: Reference isomorph not found: $REF_ISOMORPH"
    echo "AST comparison will be limited"
fi

# Create temporary directory
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=== AST Isomorphism Verification: $BASENAME ==="
echo ""

# Step 1: Transpile with cppfront (if reference doesn't exist)
if [ ! -f "$REF_CPP" ]; then
    echo "[1/5] Generating reference with cppfront..."
    "$CPPFRONT" "$INPUT_FILE" -o "$TMPDIR/ref.cpp" 2>/dev/null || {
        echo "Error: cppfront failed on $INPUT_FILE"
        exit 1
    }
    REF_CPP="$TMPDIR/ref.cpp"
else
    echo "[1/5] Using existing reference C++"
fi

# Step 2: Transpile with cppfort
echo "[2/5] Transpiling with cppfort..."
"$CPPFORT" "$INPUT_FILE" "$TMPDIR/candidate.cpp" > /dev/null 2>&1 || {
    echo "Error: cppfort failed on $INPUT_FILE"
    exit 1
}

# Step 3: Generate AST from cppfort output
echo "[3/5] Generating AST from cppfort output..."
"$CLANG++" -std=c++20 -Xclang -ast-dump -fsyntax-only "$TMPDIR/candidate.cpp" \
    > "$TMPDIR/candidate.ast.txt" 2>/dev/null || {
    echo "Error: Failed to generate AST from cppfort output"
    cat "$TMPDIR/candidate.cpp" | head -20
    exit 1
}

# Step 4: Compare ASTs
echo "[4/5] Comparing ASTs..."

if [ $VERBOSE -eq 1 ]; then
    echo "--- Reference AST (first 50 lines) ---"
    head -50 "$REF_AST"
    echo ""
    echo "--- Candidate AST (first 50 lines) ---"
    head -50 "$TMPDIR/candidate.ast.txt"
    echo ""
fi

# Simple AST comparison (line count, function count, class count)
REF_FUNCTIONS=$(grep -c "^FunctionDecl" "$REF_AST" || echo "0")
CAND_FUNCTIONS=$(grep -c "^FunctionDecl" "$TMPDIR/candidate.ast.txt" || echo "0")

REF_CLASSES=$(grep -c "^CXXRecordDecl" "$REF_AST" || echo "0")
CAND_CLASSES=$(grep -c "^CXXRecordDecl" "$TMPDIR/candidate.ast.txt" || echo "0")

REF_LINES=$(wc -l < "$REF_AST")
CAND_LINES=$(wc -l < "$TMPDIR/candidate.ast.txt")

# Calculate similarity
FUNCTION_DIFF=$((REF_FUNCTIONS - CAND_FUNCTIONS))
FUNCTION_DIFF=${FUNCTION_DIFF#-}  # Absolute value

CLASS_DIFF=$((REF_CLASSES - CAND_CLASSES))
CLASS_DIFF=${CLASS_DIFF#-}

LINE_DIFF=$((REF_LINES - CAND_LINES))
LINE_DIFF=${LINE_DIFF#-}

# Thresholds for passing
MAX_FUNCTION_DIFF=5
MAX_CLASS_DIFF=5
MAX_LINE_DIFF=100

# Step 5: Result
echo "[5/5] Result"
echo ""
echo "Reference: $REF_FUNCTIONS functions, $REF_CLASSES classes, $REF_LINES AST lines"
echo "Candidate: $CAND_FUNCTIONS functions, $CAND_CLASSES classes, $CAND_LINES AST lines"
echo "Differences: $FUNCTION_DIFF functions, $CLASS_DIFF classes, $LINE_DIFF lines"

if [ $FUNCTION_DIFF -le $MAX_FUNCTION_DIFF ] && \
   [ $CLASS_DIFF -le $MAX_CLASS_DIFF ] && \
   [ $LINE_DIFF -le $MAX_LINE_DIFF ]; then
    echo ""
    echo "✓ AST verification PASSED"
    exit 0
else
    echo ""
    echo "✗ AST verification FAILED"
    echo "  Thresholds: functions≤$MAX_FUNCTION_DIFF, classes≤$MAX_CLASS_DIFF, lines≤$MAX_LINE_DIFF"
    exit 1
fi
