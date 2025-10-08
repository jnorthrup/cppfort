#!/bin/bash
count=0
for file in regression-tests/*.cpp2; do
  result=$(./src/stage0/build/stage0_cli transpile "$file" /tmp/test.cpp 2>&1)
  if echo "$result" | grep -q "FATAL:"; then
    echo "FILE: $file"
    echo "$result" | grep -A2 "FATAL:"
    echo "---"
    ((count++))
    if [ $count -ge 10 ]; then
      break
    fi
  fi
done
