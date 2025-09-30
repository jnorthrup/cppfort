#!/bin/bash
cd "$(dirname "$0")"
CPPFRONT="../build/stage0_cli transpile"
LOG="regression_log.txt"
TEC_DIR="../tools/stage1"

rm -f "$LOG"
echo "Regression test log - $(date)" > "$LOG"
echo "Using cppfront: $CPPFRONT" >> "$LOG"
num_tests=0
num_fail=0

ORDER_SCRIPT="$TEC_DIR/order_regressions_by_error.py"
ORDER_OUTPUT=""
if [ -x "$ORDER_SCRIPT" ]; then
  ORDER_OUTPUT="$ORDER_SCRIPT --tests-dir . --log ../regression_stage1_full_log.txt"
  ORDERED_LIST="$($ORDER_SCRIPT --tests-dir . --log ../regression_stage1_full_log.txt 2>/dev/null)"
else
  ORDERED_LIST=""
fi

if [ -z "$ORDERED_LIST" ]; then
  ORDERED_LIST="$(printf '%s\n' *.cpp2 | sort)"
fi

for file in $ORDERED_LIST; do
  [ -f "$file" ] || continue
  num_tests=$((num_tests + 1))
  base="${file%.cpp2}"
  echo "Testing $file" >> "$LOG"
  output_file="output_${base}.txt"
  # Transpile
  if $CPPFRONT "$file" "${base}.cpp" >> "$LOG" 2>&1; then
    echo "  Transpile OK" >> "$LOG"
    # Compile
    if g++ -std=c++20 -O0 -g -I../include -o "$base" "${base}.cpp" >> "$LOG" 2>&1; then
      echo "  Compile OK" >> "$LOG"
      # Run
      if ./$base > "$output_file" 2>&1; then
        echo "  Run OK" >> "$LOG"
        expected="test-results/${base}.output"
        if [ -f "$expected" ]; then
          if diff "$output_file" "$expected" > /dev/null; then
            echo "  Output matches expected" >> "$LOG"
          else
            echo "  Output does not match expected" >> "$LOG"
            num_fail=$((num_fail + 1))
          fi
        else
          echo "  No expected output to compare" >> "$LOG"
        fi
      else
        echo "  Run FAILED (exit code $?)" >> "$LOG"
        num_fail=$((num_fail + 1))
      fi
    else
      echo "  Compile FAILED" >> "$LOG"
      num_fail=$((num_fail + 1))
    fi
  else
    echo "  Transpile FAILED" >> "$LOG"
    num_fail=$((num_fail + 1))
  fi
  echo "" >> "$LOG"
  rm -f "${base}.cpp" "$base" "$output_file"
done

echo "Total tests: $num_tests" >> "$LOG"
echo "Failures: $num_fail" >> "$LOG"
echo "Log saved to $LOG"
