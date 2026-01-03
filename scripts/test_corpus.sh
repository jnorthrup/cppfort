#!/usr/bin/env bash
# Standalone corpus test runner - no external dependencies
# Tests: transpile -> compile -> run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$PROJECT_ROOT/build}"
CPPFORT_BIN="$BUILD_DIR/src/cppfort"
CORPUS_DIR="$PROJECT_ROOT/corpus"
WORK_DIR="/tmp/cppfort_tests"

# Timeouts
TRANSPILE_TIMEOUT=${TRANSPILE_TIMEOUT:-15}
COMPILE_TIMEOUT=${COMPILE_TIMEOUT:-30}
RUN_TIMEOUT=${RUN_TIMEOUT:-5}

# Output format
CSV_OUTPUT="${CSV_OUTPUT:-$PROJECT_ROOT/test_results.csv}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Counters
PASS=0
FAIL=0
SKIP=0

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

echo "test,transpile,compile,run,status" > "$CSV_OUTPUT"

for cpp2file in "$CORPUS_DIR/inputs/"*.cpp2; do
    name=$(basename "$cpp2file" .cpp2)
    cppfile="$WORK_DIR/${name}.cpp"
    binfile="$WORK_DIR/${name}"
    
    transpile_ok=0
    compile_ok=0
    run_ok=0
    status="FAIL"
    
    # Stage 1: Transpile
    if timeout "$TRANSPILE_TIMEOUT" "$CPPFORT_BIN" "$cpp2file" "$cppfile" 2>/dev/null; then
        transpile_ok=1
        
        # Stage 2: Compile
        if timeout "$COMPILE_TIMEOUT" clang++ -std=c++20 -I"$PROJECT_ROOT/include" -o "$binfile" "$cppfile" 2>/dev/null; then
            compile_ok=1
            
            # Stage 3: Run
            if timeout "$RUN_TIMEOUT" "$binfile" 2>/dev/null; then
                run_ok=1
                status="PASS"
            fi
        fi
    fi
    
    echo "$name,$transpile_ok,$compile_ok,$run_ok,$status" >> "$CSV_OUTPUT"
    
    if [ "$status" = "PASS" ]; then
        echo -e "${GREEN}PASS${NC}: $name"
        ((PASS++))
    else
        echo -e "${RED}FAIL${NC}: $name (T:$transpile_ok C:$compile_ok R:$run_ok)"
        ((FAIL++))
    fi
done

TOTAL=$((PASS + FAIL))
PCT=$(echo "scale=1; $PASS * 100 / $TOTAL" | bc)

echo ""
echo "=============================="
echo -e "Passed: ${GREEN}$PASS${NC} / $TOTAL ($PCT%)"
echo -e "Failed: ${RED}$FAIL${NC}"
echo "Results: $CSV_OUTPUT"
