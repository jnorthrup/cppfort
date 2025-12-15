#!/bin/bash
# Batch extract AST isomorphs from all reference AST dumps
#
# Usage: ./tools/batch_extract_isomorphs.sh [--limit N] [--parallel N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

AST_DIR="$PROJECT_ROOT/corpus/reference_ast"
ISOMORPH_DIR="$PROJECT_ROOT/corpus/isomorphs"
EXTRACTOR="$PROJECT_ROOT/tools/extract_ast_isomorphs.py"

LIMIT=""
PARALLEL=4

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--limit N] [--parallel N]"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$ISOMORPH_DIR"

# Find all .ast.txt files
ast_files=("$AST_DIR"/*.ast.txt)
total_files=${#ast_files[@]}

if [[ -n "$LIMIT" ]]; then
    ast_files=("${ast_files[@]:0:$LIMIT}")
    total_files=$LIMIT
fi

echo "=== Batch Extracting AST Isomorphs ==="
echo "AST directory: $AST_DIR"
echo "Output directory: $ISOMORPH_DIR"
echo "Total files: $total_files"
echo "Parallel workers: $PARALLEL"
echo

# Function to process a single AST file
process_ast() {
    local ast_file="$1"
    local index="$2"
    local total="$3"

    local basename=$(basename "$ast_file" .ast.txt)
    local output_file="$ISOMORPH_DIR/$basename.isomorph.json"

    if python3 "$EXTRACTOR" --ast "$ast_file" --output "$output_file" 2>&1 | grep -q "Writing"; then
        local count=$(python3 -c "import json; data=json.load(open('$output_file')); print(data['total_isomorphs'])")
        echo "[$index/$total] $basename: $count isomorphs"
        return 0
    else
        echo "[$index/$total] $basename: FAILED"
        return 1
    fi
}

export -f process_ast
export EXTRACTOR
export ISOMORPH_DIR

# Process files in parallel
processed=0
failed=0

printf '%s\n' "${ast_files[@]}" | \
    xargs -n 1 -P "$PARALLEL" -I {} bash -c \
    'process_ast "{}" "$((++processed))" "'"$total_files"'"' || true

# Generate summary
echo
echo "=== Isomorph Extraction Summary ==="

total_isomorphs=0
for json_file in "$ISOMORPH_DIR"/*.isomorph.json; do
    if [[ -f "$json_file" ]]; then
        count=$(python3 -c "import json; data=json.load(open('$json_file')); print(data['total_isomorphs'])" 2>/dev/null || echo "0")
        total_isomorphs=$((total_isomorphs + count))
    fi
done

completed=$(ls -1 "$ISOMORPH_DIR"/*.isomorph.json 2>/dev/null | wc -l | tr -d ' ')

echo "Files processed: $completed/$total_files"
echo "Total isomorphs extracted: $total_isomorphs"
echo "Average per file: $((total_isomorphs / completed))"
echo

if [[ $completed -eq $total_files ]]; then
    echo "✓ All files processed successfully"
    exit 0
else
    echo "✗ Some files failed to process"
    exit 1
fi
