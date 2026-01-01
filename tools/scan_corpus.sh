#!/bin/bash
# Scan all corpus files and categorize results

CORPUS_DIR="third_party/cppfront/regression-tests"
OUTPUT="corpus_scan_results.txt"

echo "=== Corpus Scan Results ===" > "$OUTPUT"
echo "Date: $(date)" >> "$OUTPUT"
echo "" >> "$OUTPUT"

total=0
pass=0
fail=0
segfault=0

for file in "$CORPUS_DIR"/*.cpp2; do
  if [ ! -f "$file" ]; then continue; fi

  total=$((total + 1))
  basename=$(basename "$file" .cpp2)
  category=$(echo "$basename" | cut -d'-' -f1)

  echo -n "[$total] $basename... " >&2

  output=$(timeout 3 ./build/src/cppfort "$file" /tmp/test_scan.cpp 2>&1)
  exit_code=$?

  if [ $exit_code -eq 0 ]; then
    pass=$((pass + 1))
    echo "PASS" >&2
    echo "PASS: $basename ($category)" >> "$OUTPUT"
  elif echo "$output" | grep -q "Segmentation fault"; then
    segfault=$((segfault + 1))
    echo "SEGFAULT" >&2
    echo "SEGFAULT: $basename ($category)" >> "$OUTPUT"
  else
    fail=$((fail + 1))
    echo "FAIL (exit $exit_code)" >&2
    echo "FAIL: $basename ($category) - exit $exit_code" >> "$OUTPUT"
  fi
done

echo "" >> "$OUTPUT"
echo "=== Summary ===" >> "$OUTPUT"
echo "Total: $total" >> "$OUTPUT"
echo "Pass: $pass" >> "$OUTPUT"
echo "Fail: $fail" >> "$OUTPUT"
echo "Segfault: $segfault" >> "$OUTPUT"

echo "" >&2
echo "=== Complete ===" >&2
cat "$OUTPUT"
