#!/bin/bash
# Process corpus files and track results

BUILD_DIR="build"
CSV="$BUILD_DIR/corpus_results.csv"
CORPUS_DIR="third_party/cppfront/regression-tests"

echo "file,status,loss_score,errors,time_ms" > "$CSV"

# Files that passed in regression tests
FILES=(
  "pure2-deducing-pointers-error.cpp2"
  "pure2-deduction-1-error.cpp2"
  "pure2-deduction-2-error.cpp2"
  "pure2-bounds-safety-pointer-arithmetic-error.cpp2"
  "pure2-bugfix-for-bad-decltype-error.cpp2"
  "pure2-bugfix-for-invalid-alias-error.cpp2"
  "pure2-bugfix-for-namespace-error.cpp2"
  "mixed-allcpp1-hello.cpp2"
)

for file in "${FILES[@]}"; do
  filepath="$CORPUS_DIR/$file"
  if [ ! -f "$filepath" ]; then
    echo "SKIP: $file not found"
    continue
  fi

  basename=$(basename "$file" .cpp2)
  echo -n "Testing $basename... "

  START=$(python3 -c "import time; print(int(time.time()*1000))")
  timeout 5 ./build/src/cppfort "$filepath" "$BUILD_DIR/cppfort-$basename.cpp" 2>/dev/null >/dev/null
  EXIT_CODE=$?
  END=$(python3 -c "import time; print(int(time.time()*1000))")
  TIME=$((END - START))

  if [ $EXIT_CODE -eq 0 ]; then
    echo "PASS (${TIME}ms)"
    echo "$basename,PASS,0.0,,$TIME" >> "$CSV"
  else
    echo "FAIL (exit $EXIT_CODE)"
    echo "$basename,FAIL,1.0,\"exit $EXIT_CODE\",$TIME" >> "$CSV"
  fi
done

echo ""
echo "Results:"
cat "$CSV"
