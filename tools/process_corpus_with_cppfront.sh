#!/bin/bash
# Process entire cppfront corpus to generate reference C++1 outputs and Clang AST dumps
#
# Usage: ./tools/process_corpus_with_cppfront.sh [--limit N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CPPFRONT="$PROJECT_ROOT/third_party/cppfront/source/cppfront"
CORPUS_DIR="$PROJECT_ROOT/corpus/inputs"
REFERENCE_DIR="$PROJECT_ROOT/corpus/reference"
AST_DIR="$PROJECT_ROOT/corpus/reference_ast"

LIMIT=""
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--limit N] [--verbose]"
            exit 1
            ;;
    esac
done

# Verify cppfront binary exists
if [[ ! -x "$CPPFRONT" ]]; then
    echo "Error: cppfront binary not found at $CPPFRONT"
    echo "Run: cd third_party/cppfront/source && g++-15 -std=c++20 -O2 cppfront.cpp -o cppfront"
    exit 1
fi

# Create output directories
mkdir -p "$REFERENCE_DIR"
mkdir -p "$AST_DIR"

# Find all .cpp2 files
cpp2_files=("$CORPUS_DIR"/*.cpp2)
total_files=${#cpp2_files[@]}

if [[ -n "$LIMIT" ]]; then
    cpp2_files=("${cpp2_files[@]:0:$LIMIT}")
    total_files=$LIMIT
fi

echo "=== Processing $total_files Cpp2 files with cppfront ==="
echo "Corpus directory: $CORPUS_DIR"
echo "Reference output: $REFERENCE_DIR"
echo "AST output: $AST_DIR"
echo

processed=0
failed=0
skipped=0

for cpp2_file in "${cpp2_files[@]}"; do
    basename=$(basename "$cpp2_file" .cpp2)
    reference_cpp="$REFERENCE_DIR/$basename.cpp"
    ast_text="$AST_DIR/$basename.ast.txt"

    ((processed++))

    echo -n "[$processed/$total_files] $basename.cpp2 ... "

    # Step 1: Transpile with cppfront
    if ! $CPPFRONT "$cpp2_file" > /dev/null 2>&1; then
        echo "FAILED (cppfront error)"
        ((failed++))
        continue
    fi

    # cppfront outputs to current directory, move it
    if [[ ! -f "$basename.cpp" ]]; then
        echo "FAILED (no output generated)"
        ((failed++))
        continue
    fi

    mv "$basename.cpp" "$reference_cpp"

    # Step 2: Generate Clang AST dump
    if ! clang++ -std=c++20 \
         -I"$PROJECT_ROOT/third_party/cppfront/include" \
         -Xclang -ast-dump \
         -fsyntax-only \
         "$reference_cpp" > "$ast_text" 2>&1; then

        # AST dump may fail for files with errors, but we keep the output
        if [[ $VERBOSE == true ]]; then
            echo "WARNING (AST dump has errors, kept anyway)"
        fi
    fi

    # Calculate sizes
    cpp_size=$(wc -c < "$reference_cpp" | tr -d ' ')
    ast_size=$(wc -c < "$ast_text" | tr -d ' ')

    echo "OK (C++1: ${cpp_size}B, AST: ${ast_size}B)"
done

echo
echo "=== Summary ==="
echo "Total processed: $processed"
echo "Successfully transpiled: $((processed - failed))"
echo "Failed: $failed"
echo

if [[ $failed -eq 0 ]]; then
    echo "✓ All files processed successfully"
    exit 0
else
    echo "✗ Some files failed to process"
    exit 1
fi
