#!/bin/bash
# Corpus validation scanner
# Scans all .cpp2 files and records transpilation status

CORPUS_DIR="corpus/inputs"
OUTPUT_FILE="corpus_scan_results.txt"
TRANSPILER="./build/src/cppfort"

# Counters
total=0
passed=0
failed=0

# Clear output file
> "$OUTPUT_FILE"

echo "=== Corpus Validation Scan ==="
echo "Scanning $(ls $CORPUS_DIR/*.cpp2 | wc -l) files..."
echo ""

# Process each file in sorted order
for input_file in $(ls $CORPUS_DIR/*.cpp2 | sort); do
    total=$((total + 1))
    filename=$(basename "$input_file" .cpp2)

    # Determine file type (pure2 or mixed)
    if [[ "$filename" == pure2-* ]]; then
        file_type="pure2"
    elif [[ "$filename" == mixed-* ]]; then
        file_type="mixed"
    else
        file_type="other"
    fi

    # Try to transpile
    output_file="/tmp/corpus_scan_${filename}.cpp"
    if $TRANSPILER "$input_file" "$output_file" >/dev/null 2>&1; then
        status=$?
        if [ $status -eq 0 ]; then
            echo "PASS: $filename ($file_type)" >> "$OUTPUT_FILE"
            passed=$((passed + 1))
            printf "\r[%3d/%3d] PASS: %-60s" $total 189 "$filename"
        else
            echo "FAIL: $filename ($file_type) - exit $status" >> "$OUTPUT_FILE"
            failed=$((failed + 1))
            printf "\r[%3d/%3d] FAIL: %-60s" $total 189 "$filename"
        fi
    else
        status=$?
        echo "FAIL: $filename ($file_type) - exit $status" >> "$OUTPUT_FILE"
        failed=$((failed + 1))
        printf "\r[%3d/%3d] FAIL: %-60s" $total 189 "$filename"
    fi

    # Clean up temp file
    rm -f "$output_file"
done

echo ""
echo ""
echo "=== Scan Complete ==="
echo "Total files: $total"
echo "Passed: $passed ($(( passed * 100 / total ))%)"
echo "Failed: $failed ($(( failed * 100 / total ))%)"
echo ""
echo "Results written to: $OUTPUT_FILE"
