#!/bin/bash
cd "$(dirname "$0")"
CPPFRONT="../../stage0_clean/cppfront"
LOG="regression_log.txt"
rm -f "$LOG"
echo "Regression test log - $(date)" > "$LOG"
echo "Using cppfront: $CPPFRONT" >> "$LOG"
num_tests=0
num_fail=0
for file in *.cpp2; do
  if [ -f "$file" ]; then
    num_tests=$((num_tests + 1))
    base="${file%.cpp2}"
    echo "Testing $file" >> "$LOG"
    # Transpile
    if $CPPFRONT "$file" -o "${base}.cpp" >> "$LOG" 2>&1; then
      echo "  Transpile OK" >> "$LOG"
      # Compile
      if g++ -std=c++20 -O0 -g -I../../stage0_clean/include -o "$base" "${base}.cpp" >> "$LOG" 2>&1; then
        echo "  Compile OK" >> "$LOG"
        # Run
        output_file="output_${base}.txt"
        if ./$base > "$output_file" 2>&1; then
          echo "  Run OK" >> "$LOG"
          # Check if expected output exists
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
  fi
done
echo "Total tests: $num_tests" >> "$LOG"
echo "Failures: $num_fail" >> "$LOG"
echo "Log saved to $LOG"

# Post-process: parse the regression log into structured JSON/CSV and link errors
echo "Post-processing regression results..." >> "$LOG"
TOOLS_DIR="../tools/stage1"
python3 "$TOOLS_DIR/parse_regression.py" "$LOG" >> "$LOG" 2>&1 || echo "Parser failed" >> "$LOG"
python3 "$TOOLS_DIR/link_errors_to_context.py" "$TOOLS_DIR/regression_summary.json" >> "$LOG" 2>&1 || echo "Linker failed" >> "$LOG"
echo "Post-processing complete. Outputs in $TOOLS_DIR" >> "$LOG"