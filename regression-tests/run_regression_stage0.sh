#!/bin/bash
cd "$(dirname "$0")"
CPPFRONT="../build/stage0_cli transpile"
LOG="regression_log.txt"
rm -f "$LOG"
echo "Regression test log - $(date)" > "$LOG"
echo "Using cppfront: $CPPFRONT" >> "$LOG"

# Tests that require safety features not yet implemented in stage0
SKIP_TESTS=(
    "pure2-assert-expected-not-null.cpp2"      # requires C++23 std::expected
    "pure2-assert-optional-not-null.cpp2"      # requires safe dereference
    "pure2-assert-shared-ptr-not-null.cpp2"    # requires safe dereference  
    "pure2-assert-unique-ptr-not-null.cpp2"    # requires safe dereference
    "pure2-bounds-safety-pointer-arithmetic-error.cpp2"  # requires bounds safety
    "pure2-bounds-safety-span.cpp2"            # requires bounds safety
)

is_skipped() {
    local test_file="$1"
    for skip in "${SKIP_TESTS[@]}"; do
        if [ "$test_file" = "$skip" ]; then
            return 0
        fi
    done
    return 1
}

num_tests=0
num_fail=0
num_skipped=0
for file in pure2-*.cpp2; do
  if [ -f "$file" ]; then
    if is_skipped "$file"; then
      echo "Skipping $file (requires unimplemented safety features)" >> "$LOG"
      num_skipped=$((num_skipped + 1))
      continue
    fi
    num_tests=$((num_tests + 1))
    base="${file%.cpp2}"
    echo "Testing $file" >> "$LOG"
    # Transpile
    if $CPPFRONT "$file" "${base}.cpp" >> "$LOG" 2>&1; then
      echo "  Transpile OK" >> "$LOG"
      # Compile
      if g++ -std=c++23 -O0 -g -I../../stage0_clean/include -I../include -o "$base" "${base}.cpp" >> "$LOG" 2>&1; then
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
echo "Skipped: $num_skipped" >> "$LOG"
echo "Log saved to $LOG"

# Post-process: parse the regression log into structured JSON/CSV and link errors
echo "Post-processing regression results..." >> "$LOG"
TOOLS_DIR="../tools/stage1"
python3 "$TOOLS_DIR/parse_regression.py" "$LOG" >> "$LOG" 2>&1 || echo "Parser failed" >> "$LOG"
python3 "$TOOLS_DIR/link_errors_to_context.py" "$TOOLS_DIR/regression_summary.json" >> "$LOG" 2>&1 || echo "Linker failed" >> "$LOG"
echo "Post-processing complete. Outputs in $TOOLS_DIR" >> "$LOG"
