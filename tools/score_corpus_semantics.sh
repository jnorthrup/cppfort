#!/bin/bash
# Score semantic loss for the entire corpus
# Usage: ./tools/score_corpus_semantics.sh [--limit N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Tools
CPPFORT_BIN="$PROJECT_ROOT/build/src/cppfort"
EXTRACT_TOOL="$PROJECT_ROOT/tools/extract_ast_isomorphs.py"
TAG_TOOL="$PROJECT_ROOT/tools/tag_mlir_regions.py"
SCORE_TOOL="$PROJECT_ROOT/tools/score_semantic_loss.py"

# Directories
CORPUS_DIR="$PROJECT_ROOT/corpus/inputs"
REF_AST_DIR="$PROJECT_ROOT/corpus/reference_ast"
WORK_DIR="$PROJECT_ROOT/build/semantic_scoring_work"
RESULTS_FILE="$PROJECT_ROOT/build/semantic_loss_results.csv"

# Check dependencies
if [[ ! -f "$CPPFORT_BIN" ]]; then
    echo "Error: cppfort binary not found at $CPPFORT_BIN"
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

# Initialize CSV if not exists
if [[ ! -f "$RESULTS_FILE" ]]; then
    echo "file,status,loss,struct_dist,type_dist,op_dist,ref_patterns,cand_patterns,matched" > "$RESULTS_FILE"
fi

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

echo "=== Scoring Semantic Loss for ${#cpp2_files[@]} files (Skipped: $SKIP) ==="
echo "Work dir: $WORK_DIR"
echo "Results: $RESULTS_FILE"

processed=$SKIP
failed=0
total_loss=0

for cpp2_file in "${cpp2_files[@]}"; do
    basename=$(basename "$cpp2_file" .cpp2)
    ref_ast="$REF_AST_DIR/$basename.ast.txt"
    
    ((processed++))
    echo -n "[$processed/$total_files] $basename ... "
    
    # Check if reference AST exists
    if [[ ! -f "$ref_ast" ]]; then
        echo "SKIP (no reference AST)"
        echo "$basename,SKIP,,,,,,,," >> "$RESULTS_FILE"
        continue
    fi
    
    # 1. Process Reference (if not cached)
    ref_json="$WORK_DIR/$basename.ref.json"
    ref_tagged="$WORK_DIR/$basename.ref.tagged.json"
    
    if [[ ! -f "$ref_tagged" ]]; then
        # Extract isomorphs
        if ! python3 "$EXTRACT_TOOL" --ast "$ref_ast" --output "$ref_json" >/dev/null 2>&1; then
            echo "FAIL (ref extract)"
            echo "$basename,FAIL_REF_EXTRACT,,,,,,,," >> "$RESULTS_FILE"
            ((failed++))
            continue
        fi
        
        # Tag regions
        if ! python3 "$TAG_TOOL" --isomorphs "$ref_json" --output "$ref_tagged" >/dev/null 2>&1; then
            echo "FAIL (ref tag)"
            echo "$basename,FAIL_REF_TAG,,,,,,,," >> "$RESULTS_FILE"
            ((failed++))
            continue
        fi
    fi
    
    # 2. Process Candidate
    cand_cpp="$WORK_DIR/$basename.cand.cpp"
    cand_ast="$WORK_DIR/$basename.cand.ast.txt"
    cand_json="$WORK_DIR/$basename.cand.json"
    cand_tagged="$WORK_DIR/$basename.cand.tagged.json"
    loss_json="$WORK_DIR/$basename.loss.json"
    
    # Transpile
    if ! "$CPPFORT_BIN" "$cpp2_file" "$cand_cpp" >/dev/null 2>&1; then
        echo "FAIL (transpile)"
        echo "$basename,FAIL_TRANSPILE,1.0,,,,,," >> "$RESULTS_FILE"
        ((failed++))
        continue
    fi
    
    # Generate AST dump
    # Note: We use -fsyntax-only to speed it up and avoid linking
    # We ignore errors because we want the AST even if partial
    clang++ -std=c++20 -I"$PROJECT_ROOT/include" -Xclang -ast-dump -fsyntax-only "$cand_cpp" > "$cand_ast" 2>&1 || true
    
    # Extract isomorphs
    if ! python3 "$EXTRACT_TOOL" --ast "$cand_ast" --output "$cand_json" >/dev/null 2>&1; then
        echo "FAIL (cand extract)"
        echo "$basename,FAIL_CAND_EXTRACT,1.0,,,,,," >> "$RESULTS_FILE"
        ((failed++))
        continue
    fi

    # Tag regions (Candidate)
    if ! python3 "$TAG_TOOL" --isomorphs "$cand_json" --output "$cand_tagged" >/dev/null 2>&1; then
        echo "FAIL (cand tag)"
        echo "$basename,FAIL_CAND_TAG,1.0,,,,,," >> "$RESULTS_FILE"
        ((failed++))
        continue
    fi
    
    # 3. Score
    # Note: Scorer expects keys "tagged_isomorphs" in both files if we use tagging tool on both
    # But checking score_semantic_loss.py, it looks for "isomorphs" key in candidate if not tagged?
    # Actually, the scorer assumes nested 'ast_pattern' structure which comes from the tagger.
    if ! python3 "$SCORE_TOOL" --reference "$ref_tagged" --candidate "$cand_tagged" --output "$loss_json" >/dev/null 2>&1; then
        echo "FAIL (scoring)"
        echo "$basename,FAIL_SCORING,1.0,,,,,," >> "$RESULTS_FILE"
        ((failed++))
        continue
    fi
    
    # Extract metrics from JSON using python one-liner
    read loss s_dist t_dist o_dist ref_pat cand_pat match <<< $(python3 -c "
import json
with open('$loss_json') as f:
    m = json.load(f)['metrics']
    print(f\"{m['combined_loss']} {m['structural_distance']} {m['type_distance']} {m['operation_distance']} {m['reference_isomorphs']} {m['candidate_isomorphs']} {m['matched_patterns']}\")
")
    
    echo "$basename,OK,$loss,$s_dist,$t_dist,$o_dist,$ref_pat,$cand_pat,$match" >> "$RESULTS_FILE"
    
    # Color output based on loss
    if (( $(echo "$loss < 0.1" | bc -l) )); then
        echo "OK (Loss: $loss)"
    elif (( $(echo "$loss < 0.5" | bc -l) )); then
        echo "WARN (Loss: $loss)"
    else
        echo "HIGH (Loss: $loss)"
    fi
    
    total_loss=$(echo "$total_loss + $loss" | bc -l)

done

# Calculate average
if [[ -f "$PROJECT_ROOT/tools/summarize_loss.py" ]]; then
    python3 "$PROJECT_ROOT/tools/summarize_loss.py" "$RESULTS_FILE"
else
    echo "Results saved to: $RESULTS_FILE"
fi

exit 0
