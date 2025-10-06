#!/bin/bash
LOGFILE="regression_log.txt"
echo "Regression test log - $(date)" > $LOGFILE
echo "Using cppfront: ../build/stage0_cli transpile" >> $LOGFILE

TOTAL=0
FAILED=0

for test in *.cpp2; do
    echo "Testing $test" >> $LOGFILE
    ((TOTAL++))
    
    # Try to load the file and transpile
    if ../build/stage0_cli transpile "$test" "/tmp/out_${test%.cpp2}.cpp" >> $LOGFILE 2>&1; then
        echo "  Transpile OK" >> $LOGFILE
    else
        echo "  Transpile FAILED" >> $LOGFILE
        ((FAILED++))
    fi
    echo "" >> $LOGFILE
done

echo "Total tests: $TOTAL" >> $LOGFILE
echo "Failures: $FAILED" >> $LOGFILE
