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
  
  # Check if this is an error test (should fail with errors)
  is_error_test=false
  if [[ "$basename" == *"-error" ]]; then
    is_error_test=true
  fi

  echo -n "[$total] $basename... " >&2

  output=$(timeout 3 ./build/src/cppfort "$file" build/test_scan.cpp 2>&1)
  exit_code=$?

  if [ $exit_code -eq 0 ]; then
    if [ "$is_error_test" = true ]; then
      # Error test should have failed but didn't
      fail=$((fail + 1))
      echo "FAIL (should error)" >&2
      echo "FAIL: $basename ($category) - should have produced error" >> "$OUTPUT"
    else
      pass=$((pass + 1))
      echo "PASS" >&2
      echo "PASS: $basename ($category)" >> "$OUTPUT"
    fi
  elif [ $exit_code -eq 139 ] || [ $exit_code -eq 134 ] || [ $exit_code -eq 136 ] || echo "$output" | grep -qi "segmentation fault\|abort\|signal"; then
    # 139 = SIGSEGV, 134 = SIGABRT, 136 = SIGFPE
    segfault=$((segfault + 1))
    echo "SEGFAULT (exit $exit_code)" >&2
    echo "SEGFAULT: $basename ($category) - exit $exit_code" >> "$OUTPUT"
  else
    if [ "$is_error_test" = true ]; then
      # Error test correctly produced an error
      pass=$((pass + 1))
      echo "PASS (expected error)" >&2
      echo "PASS: $basename ($category) - expected error" >> "$OUTPUT"
    else
      fail=$((fail + 1))
      echo "FAIL (exit $exit_code)" >&2
      echo "FAIL: $basename ($category) - exit $exit_code" >> "$OUTPUT"
    fi
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
