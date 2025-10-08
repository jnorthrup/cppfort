#!/bin/bash

# Analyze fatal error patterns

echo "Analyzing fatal error patterns..."
echo

declare -A error_counts

for file in regression-tests/*.cpp2; do
    output=$(./src/stage0/build/stage0_cli transpile "$file" /tmp/test_out.cpp 2>&1)

    if echo "$output" | grep -q "FATAL:"; then
        error=$(echo "$output" | grep "FATAL:" | head -1)
        ((error_counts["$error"]++))
    fi
done

echo "Fatal error frequency:"
for error in "${!error_counts[@]}"; do
    echo "  [${error_counts[$error]}] $error"
done | sort -rn -t']' -k1
