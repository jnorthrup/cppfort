#!/bin/bash

# Measure orbit detection accuracy on regression tests
# Reports: success rate, fatal errors, warnings

total=0
success=0
fatal=0
warnings=0

echo "Testing orbit accuracy on regression tests..."
echo

for file in regression-tests/*.cpp2; do
    ((total++))

    output=$(./src/stage0/build/stage0_cli transpile "$file" /tmp/test_out.cpp 2>&1)

    if echo "$output" | grep -q "FATAL:"; then
        ((fatal++))
        # echo "FATAL: $file"
    elif echo "$output" | grep -q "WARNING:"; then
        ((warnings++))
        ((success++))
        # echo "WARN:  $file"
    else
        ((success++))
        # echo "OK:    $file"
    fi
done

echo "Results:"
echo "  Total files:    $total"
echo "  Successful:     $success"
echo "  Fatal errors:   $fatal"
echo "  Warnings:       $warnings"
echo

accuracy=$(awk "BEGIN {printf \"%.1f\", ($success * 100.0) / $total}")
echo "Orbit accuracy: ${accuracy}%"
