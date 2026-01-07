#!/bin/bash
# Batch compare clang AST dumps for the entire corpus
# Usage: ./tools/batch_compare_ast_dumps.sh [--limit N] [--skip N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Tools
COMPARE_TOOL="$PROJECT_ROOT/tools/compare_ast_dumps.py"
CPPFORT_BIN="$PROJECT_ROOT/build/src/cppfort"

# Directories
CORPUS_DIR="$PROJECT_ROOT/corpus/inputs"
REF_AST_DIR="$PROJECT_ROOT/corpus/reference_ast"
WORK_DIR="$PROJECT_ROOT/build/ast_comparison_work"
RESULTS_FILE="$PROJECT_ROOT/build/ast_comparison_results.csv"

# Check dependencies
if [[ ! -f "$CPPFORT_BIN" ]]; then
    echo "Error: cppfort binary not found at $CPPFORT_BIN"
    exit 1
fi

if [[ ! -f "$COMPARE_TOOL" ]]; then
    echo "Error: comparison tool not found at $COMPARE_TOOL"
    exit 1
fi

mkdir -p "$WORK_DIR"

# Parse arguments
LIMIT=""
SKIP=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --skip)
            SKIP="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Initialize CSV
echo "file,status,similarity,loss,ref_nodes,cand_nodes,matched,missing,extra,mismatches" > "$RESULTS_FILE"

# Find files
cpp2_files=($CORPUS_DIR/*.cpp2)
total_files=${#cpp2_files[@]}

# Apply skip
if [[ $SKIP -gt 0 ]]; then
    cpp2_files=("${cpp2_files[@]:$SKIP}")
fi

if [[ -n "$LIMIT" ]]; then
    cpp2_files=("${cpp2_files[@]:0:$LIMIT}")
fi

echo "=== Comparing AST Dumps for ${#cpp2_files[@]} files (Skipped: $SKIP) ==="
echo "Work dir: $WORK_DIR"
echo "Results: $RESULTS_FILE"

processed=$SKIP
failed=0
total_similarity=0
total_loss=0

for cpp2_file in "${cpp2_files[@]}"; do
    basename=$(basename "$cpp2_file" .cpp2)
    ref_ast="$REF_AST_DIR/$basename.ast.txt"
    output_json="$WORK_DIR/$basename.ast_comparison.json"

    ((processed++))
    echo -n "[$processed/$total_files] $basename ... "

    # Check if reference AST exists
    if [[ ! -f "$ref_ast" ]]; then
        echo "SKIP (no reference AST)"
        echo "$basename,SKIP,,,,,,,,," >> "$RESULTS_FILE"
        continue
    fi

    # Run comparison
    if python3 "$COMPARE_TOOL" \
        --cpp2 "$cpp2_file" \
        --ref-ast "$ref_ast" \
        --cppfort-bin "$CPPFORT_BIN" \
        --output "$output_json" \
        >/dev/null 2>&1; then

        # Extract metrics from JSON
        read similarity loss ref_nodes cand_nodes matched missing extra mismatches <<< $(python3 -c "
import json
with open('$output_json') as f:
    m = json.load(f)['metrics']
    print(f\"{m['structural_similarity']} {m['semantic_loss']} {m['ref_total_nodes']} {m['cand_total_nodes']} {m['matched_nodes']} {m['missing_nodes']} {m['extra_nodes']} {m['kind_mismatches']}\")
")

        echo "$basename,OK,$similarity,$loss,$ref_nodes,$cand_nodes,$matched,$missing,$extra,$mismatches" >> "$RESULTS_FILE"

        # Color output based on loss
        if (( $(echo "$loss == 0.0" | bc -l 2>/dev/null || echo "0") )); then
            echo "PERFECT (Loss: $loss)"
        elif (( $(echo "$loss < 0.1" | bc -l 2>/dev/null || echo "0") )); then
            echo "OK (Loss: $loss)"
        elif (( $(echo "$loss < 0.5" | bc -l 2>/dev/null || echo "0") )); then
            echo "WARN (Loss: $loss)"
        else
            echo "HIGH (Loss: $loss)"
        fi

        total_similarity=$(echo "$total_similarity + $similarity" | bc -l 2>/dev/null || echo "$total_similarity")
        total_loss=$(echo "$total_loss + $loss" | bc -l 2>/dev/null || echo "$total_loss")
    else
        echo "FAIL (comparison)"
        echo "$basename,FAIL,,,,,,,,," >> "$RESULTS_FILE"
        ((failed++))
    fi
done

# Calculate average
processed_count=$((processed - SKIP))
if [[ $processed_count -gt 0 ]]; then
    avg_similarity=$(echo "scale=4; $total_similarity / $processed_count" | bc -l 2>/dev/null || echo "0")
    avg_loss=$(echo "scale=4; $total_loss / $processed_count" | bc -l 2>/dev/null || echo "0")

    echo ""
    echo "=== Summary ==="
    echo "Processed: $processed_count"
    echo "Failed: $failed"
    echo "Average structural similarity: $avg_similarity"
    echo "Average semantic loss: $avg_loss"

    # Append summary to CSV
    echo "" >> "$RESULTS_FILE"
    echo "SUMMARY,$processed_count,$avg_similarity,$avg_loss" >> "$RESULTS_FILE"
fi

exit 0
