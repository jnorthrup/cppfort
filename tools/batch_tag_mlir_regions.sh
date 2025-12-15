#!/bin/bash
# Batch tag isomorphs with MLIR region patterns
#
# Usage: ./tools/batch_tag_mlir_regions.sh [--limit N] [--parallel N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ISOMORPH_DIR="$PROJECT_ROOT/corpus/isomorphs"
TAGGED_DIR="$PROJECT_ROOT/corpus/tagged"
TAGGER="$PROJECT_ROOT/tools/tag_mlir_regions.py"

LIMIT=""
PARALLEL=8

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
mkdir -p "$TAGGED_DIR"

# Find all .isomorph.json files
isomorph_files=("$ISOMORPH_DIR"/*.isomorph.json)
total_files=${#isomorph_files[@]}

if [[ -n "$LIMIT" ]]; then
    isomorph_files=("${isomorph_files[@]:0:$LIMIT}")
    total_files=$LIMIT
fi

echo "=== Batch Tagging Isomorphs with MLIR Regions ==="
echo "Isomorph directory: $ISOMORPH_DIR"
echo "Output directory: $TAGGED_DIR"
echo "Total files: $total_files"
echo "Parallel workers: $PARALLEL"
echo

# Function to process a single isomorph file
process_isomorph() {
    local isomorph_file="$1"
    local basename=$(basename "$isomorph_file" .isomorph.json)
    local output_file="$TAGGED_DIR/$basename.tagged.json"

    if python3 "$TAGGER" --isomorphs "$isomorph_file" --output "$output_file" 2>&1 | grep -q "Writing"; then
        local count=$(python3 -c "import json; data=json.load(open('$output_file')); print(data['total_tagged'])")
        echo "$basename: $count tagged"
        return 0
    else
        echo "$basename: FAILED"
        return 1
    fi
}

export -f process_isomorph
export TAGGER
export TAGGED_DIR

# Process files in parallel
printf '%s\n' "${isomorph_files[@]}" | \
    xargs -n 1 -P "$PARALLEL" -I {} bash -c 'process_isomorph "{}"'

# Generate summary
echo
echo "=== Tagging Summary ==="

total_tagged=0
for json_file in "$TAGGED_DIR"/*.tagged.json; do
    if [[ -f "$json_file" ]]; then
        count=$(python3 -c "import json; data=json.load(open('$json_file')); print(data['total_tagged'])" 2>/dev/null || echo "0")
        total_tagged=$((total_tagged + count))
    fi
done

completed=$(ls -1 "$TAGGED_DIR"/*.tagged.json 2>/dev/null | wc -l | tr -d ' ')

echo "Files processed: $completed/$total_files"
echo "Total tagged isomorphs: $total_tagged"
echo "Average per file: $((total_tagged / completed))"
echo

if [[ $completed -eq $total_files ]]; then
    echo "✓ All files tagged successfully"
    exit 0
else
    echo "✗ Some files failed to tag"
    exit 1
fi
