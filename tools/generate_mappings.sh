#!/bin/bash
# Generate semantic mappings from cpp2 regression tests

set -e

CPPFRONT="third_party/cppfront/source/cppfront"
REGRESSION_DIR="third_party/cppfront/regression-tests"
OUTPUT_DIR="corpus/ast_mappings/generated"
CLANG_FLAGS="-std=c++20 -I third_party/cppfront/include"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Generating semantic mappings from cpp2 regression tests..."
echo "Output directory: $OUTPUT_DIR"
echo ""

# Process first 10 files as a test batch
count=0
for cpp2_file in "$REGRESSION_DIR"/*.cpp2; do
    basename=$(basename "$cpp2_file" .cpp2)

    # Skip files that are known to be problematic
    if [[ $basename == *"-error"* ]]; then
        echo "Skipping error test: $basename"
        continue
    fi

    echo "Processing: $basename.cpp2"

    # Step 1: Generate C++1 with cppfront
    if ! $CPPFRONT "$cpp2_file" -o "/tmp/$basename.cpp" 2>&1; then
        echo "  ⚠️  cppfront failed for $basename"
        continue
    fi

    # Step 2: Generate Clang AST
    if ! clang++ $CLANG_FLAGS -Xclang -ast-dump -fsyntax-only "/tmp/$basename.cpp" > "/tmp/$basename.ast" 2>&1; then
        echo "  ⚠️  clang AST generation failed for $basename"
        continue
    fi

    # Step 3: Extract function declarations (simple pattern extraction)
    echo "=== Cpp2 Source ===" > "$OUTPUT_DIR/$basename.mapping"
    cat "$cpp2_file" >> "$OUTPUT_DIR/$basename.mapping"
    echo "" >> "$OUTPUT_DIR/$basename.mapping"

    echo "=== Generated C++1 ===" >> "$OUTPUT_DIR/$basename.mapping"
    cat "/tmp/$basename.cpp" >> "$OUTPUT_DIR/$basename.mapping"
    echo "" >> "$OUTPUT_DIR/$basename.mapping"

    # Extract key AST patterns
    echo "=== AST Patterns ===" >> "$OUTPUT_DIR/$basename.mapping"
    grep -E "FunctionDecl.*(name|decorate|main)" "/tmp/$basename.ast" | head -5 >> "$OUTPUT_DIR/$basename.mapping"

    # Look for variable declarations
    grep -E "VarDecl.*s.*std::string" "/tmp/$basename.ast" | head -3 >> "$OUTPUT_DIR/$basename.mapping"

    # Look for function calls
    grep -E "CallExpr.*decorate|CallExpr.*name" "/tmp/$basename.ast" | head -3 >> "$OUTPUT_DIR/$basename.mapping"

    echo "✓ Generated mapping: $OUTPUT_DIR/$basename.mapping"

    count=$((count + 1))
    if [ $count -ge 5 ]; then
        break
    fi
done

echo ""
echo "Processed $count files successfully"
echo "Mappings saved to $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Review generated mappings"
echo "2. Extract common patterns"
echo "3. Create MLIR mapping rules"
echo "4. Process remaining regression tests"
