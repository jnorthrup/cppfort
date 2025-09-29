#!/bin/bash
cd "$(dirname "$0")"

# Stage0: Our internal C++ emitter
STAGE0_CMD="../build/src/stage0/stage0 transpile"

# Stage1: Cpp2→C++ transpiler (placeholder - will fail until implemented)
STAGE1_CMD="../build/src/stage1/stage1 transpile"

LOG="side_by_side_log.txt"
rm -f "$LOG"
echo "Side-by-side regression test log - $(date)" > "$LOG"
echo "Stage0: $STAGE0_CMD" >> "$LOG"
echo "Stage1: $STAGE1_CMD" >> "$LOG"
echo "" >> "$LOG"

num_tests=0
num_stage0_ok=0
num_stage1_ok=0
num_both_ok=0

for file in pure2-*.cpp2; do
  if [ -f "$file" ]; then
    num_tests=$((num_tests + 1))
    base="${file%.cpp2}"
    echo "Testing $file" >> "$LOG"

    # Stage0 transpile
    stage0_cpp="${base}_stage0.cpp"
    stage0_binary="${base}_stage0"
    stage0_output="output_${base}_stage0.txt"
    rm -f "$stage0_cpp" "$stage0_binary" "$stage0_output"

    echo "  Stage0 transpile..." >> "$LOG"
    if $STAGE0_CMD "$file" "$stage0_cpp" >> "$LOG" 2>&1; then
      echo "    Stage0 transpile OK" >> "$LOG"
      stage0_transpile_ok=true
    else
      echo "    Stage0 transpile FAILED" >> "$LOG"
      stage0_transpile_ok=false
    fi

    # Stage1 transpile
    stage1_cpp="${base}_stage1.cpp"
    stage1_binary="${base}_stage1"
    stage1_output="output_${base}_stage1.txt"
    rm -f "$stage1_cpp" "$stage1_binary" "$stage1_output"

    echo "  Stage1 transpile..." >> "$LOG"
    if $STAGE1_CMD "$file" "$stage1_cpp" >> "$LOG" 2>&1; then
      echo "    Stage1 transpile OK" >> "$LOG"
      stage1_transpile_ok=true
    else
      echo "    Stage1 transpile FAILED (expected until implemented)" >> "$LOG"
      stage1_transpile_ok=false
    fi

    # Compare transpiled outputs if both succeeded
    if [ "$stage0_transpile_ok" = true ] && [ "$stage1_transpile_ok" = true ]; then
      echo "  Comparing transpiled C++..." >> "$LOG"
      if diff "$stage0_cpp" "$stage1_cpp" > /dev/null; then
        echo "    C++ outputs identical" >> "$LOG"
      else
        echo "    C++ outputs differ" >> "$LOG"
        diff "$stage0_cpp" "$stage1_cpp" >> "$LOG"
      fi
    fi

    # Compile Stage0 if transpile OK
    if [ "$stage0_transpile_ok" = true ]; then
      echo "  Stage0 compile..." >> "$LOG"
      if g++ -std=c++20 -O0 -g -I../../include -o "$stage0_binary" "$stage0_cpp" >> "$LOG" 2>&1; then
        echo "    Stage0 compile OK" >> "$LOG"
        stage0_compile_ok=true
        num_stage0_ok=$((num_stage0_ok + 1))
      else
        echo "    Stage0 compile FAILED" >> "$LOG"
        stage0_compile_ok=false
      fi
    else
      stage0_compile_ok=false
    fi

    # Compile Stage1 if transpile OK
    if [ "$stage1_transpile_ok" = true ]; then
      echo "  Stage1 compile..." >> "$LOG"
      if g++ -std=c++20 -O0 -g -I../../include -o "$stage1_binary" "$stage1_cpp" >> "$LOG" 2>&1; then
        echo "    Stage1 compile OK" >> "$LOG"
        stage1_compile_ok=true
        num_stage1_ok=$((num_stage1_ok + 1))
      else
        echo "    Stage1 compile FAILED" >> "$LOG"
        stage1_compile_ok=false
      fi
    else
      stage1_compile_ok=false
    fi

    # Run both if both compiled
    if [ "$stage0_compile_ok" = true ] && [ "$stage1_compile_ok" = true ]; then
      echo "  Running both binaries..." >> "$LOG"
      num_both_ok=$((num_both_ok + 1))

      if ./$stage0_binary > "$stage0_output" 2>&1; then
        echo "    Stage0 run OK" >> "$LOG"
        stage0_run_ok=true
      else
        echo "    Stage0 run FAILED (exit $?)" >> "$LOG"
        stage0_run_ok=false
      fi

      if ./$stage1_binary > "$stage1_output" 2>&1; then
        echo "    Stage1 run OK" >> "$LOG"
        stage1_run_ok=true
      else
        echo "    Stage1 run FAILED (exit $?)" >> "$LOG"
        stage1_run_ok=false
      fi

      # Compare outputs if both ran
      if [ "$stage0_run_ok" = true ] && [ "$stage1_run_ok" = true ]; then
        if diff "$stage0_output" "$stage1_output" > /dev/null; then
          echo "    Outputs identical" >> "$LOG"
        else
          echo "    Outputs differ" >> "$LOG"
          diff "$stage0_output" "$stage1_output" >> "$LOG"
        fi
      fi

      # Collect binary metrics for graph nodes
      stage0_size=$(stat -f%z "$stage0_binary" 2>/dev/null || echo "0")
      stage1_size=$(stat -f%z "$stage1_binary" 2>/dev/null || echo "0")
      echo "    Stage0 binary size: $stage0_size bytes" >> "$LOG"
      echo "    Stage1 binary size: $stage1_size bytes" >> "$LOG"

      # Disassemble for anticheat features (placeholder)
      if command -v objdump >/dev/null 2>&1; then
        stage0_disasm="${base}_stage0.disasm"
        stage1_disasm="${base}_stage1.disasm"
        objdump -d "$stage0_binary" > "$stage0_disasm" 2>/dev/null
        objdump -d "$stage1_binary" > "$stage1_disasm" 2>/dev/null
        stage0_disasm_lines=$(wc -l < "$stage0_disasm" 2>/dev/null || echo "0")
        stage1_disasm_lines=$(wc -l < "$stage1_disasm" 2>/dev/null || echo "0")
        echo "    Stage0 disassembly lines: $stage0_disasm_lines" >> "$LOG"
        echo "    Stage1 disassembly lines: $stage1_disasm_lines" >> "$LOG"
      fi
    fi

    echo "" >> "$LOG"
    # Cleanup
    rm -f "$stage0_cpp" "$stage0_binary" "$stage0_output" "$stage1_cpp" "$stage1_binary" "$stage1_output"
  fi
done

echo "Total tests: $num_tests" >> "$LOG"
echo "Stage0 successful: $num_stage0_ok" >> "$LOG"
echo "Stage1 successful: $num_stage1_ok" >> "$LOG"
echo "Both successful: $num_both_ok" >> "$LOG"
echo "Log saved to $LOG"

# Post-process for graph nodes
echo "Post-processing for graph nodes..." >> "$LOG"
python3 "../tools/stage1/parse_side_by_side.py" "$LOG" >> "$LOG" 2>&1 || echo "Parser failed" >> "$LOG"
echo "Post-processing complete."