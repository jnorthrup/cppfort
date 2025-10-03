#!/bin/bash

# Stage 2 Differential Extraction Script
# Compiles CPP2 regression tests at multiple optimization levels
# and extracts differential patterns for analysis

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
REGRESSION_DIR="$PROJECT_ROOT/regression-tests"
OUTPUT_DIR="$BUILD_DIR/differential_output"

# Optimization levels to test
OPT_LEVELS=("-O0")

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Stage 2 Differential Extraction"
echo "================================"
echo "Project root: $PROJECT_ROOT"
echo "Build dir: $BUILD_DIR"
echo "Regression dir: $REGRESSION_DIR"
echo "Output dir: $OUTPUT_DIR"
echo

# Check if build exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory does not exist. Run cmake first."
    exit 1
fi

# Check if stage1_cli exists
if [ ! -f "$BUILD_DIR/stage1_cli" ]; then
    echo "Error: stage1_cli not found. Build the project first."
    exit 1
fi

# Find CPP2 test files
CPP2_FILES=$(find "$REGRESSION_DIR" -name "*.cpp2" | head -1) # Just test one file first

if [ -z "$CPP2_FILES" ]; then
    echo "Error: No .cpp2 files found in $REGRESSION_DIR"
    exit 1
fi

echo "Found $(echo "$CPP2_FILES" | wc -l) CPP2 test files"
echo

# Create differential analysis tool (temporary)
cat > "$OUTPUT_DIR/differential_analyzer.cpp" << 'EOF'
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "../src/stage2/asm_parser.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <binary_path>\n";
        return 1;
    }

    std::string binary_path = argv[1];

    cppfort::stage2::AsmParser parser;

    // Parse binary
    auto instructions = parser.parseFile(binary_path);

    if (instructions.empty()) {
        std::cerr << "Failed to parse instructions from binary\n";
        return 1;
    }

    // Output results
    std::cout << "Parsed " << instructions.size() << " instructions from " << binary_path << "\n";
    for (size_t i = 0; i < std::min(size_t(10), instructions.size()); ++i) {
        const auto& inst = instructions[i];
        std::cout << "  " << std::hex << inst.address << ": " << inst.opcode << " " << inst.operands << "\n";
    }
    if (instructions.size() > 10) {
        std::cout << "  ... and " << (instructions.size() - 10) << " more\n";
    }

    return 0;
}
EOF

# Compile the analyzer
echo "Compiling differential analyzer..."
g++ -std=c++20 -I"$PROJECT_ROOT/include" -I"$PROJECT_ROOT/src" \
    -I"$PROJECT_ROOT/src/stage0" -I"$PROJECT_ROOT/src/attestation" \
    "$OUTPUT_DIR/differential_analyzer.cpp" \
    -L"$BUILD_DIR/src/stage2" -lstage2 \
    -o "$OUTPUT_DIR/differential_analyzer"

# Process each CPP2 file
for cpp2_file in $CPP2_FILES; do
    filename=$(basename "$cpp2_file" .cpp2)
    echo "Processing $filename..."

    # Compile at each optimization level
    declare -A binaries
    for opt in "${OPT_LEVELS[@]}"; do
        output_cpp="$OUTPUT_DIR/${filename}_${opt}.cpp"
        output_binary="$OUTPUT_DIR/${filename}_${opt}"

        echo "  Compiling with $opt..."

        # Convert CPP2 to C++
        if ! "$BUILD_DIR/stage1_cli" "$cpp2_file" "$output_cpp" 2>/dev/null; then
            echo "    Failed to convert CPP2 to C++"
            continue
        fi

        # Compile C++ to binary
        if ! clang++ "$opt" -o "$output_binary" "$output_cpp" 2>/dev/null; then
            echo "    Failed to compile C++ to binary"
            continue
        fi

        binaries[$opt]="$output_binary"

        # Test ASM parsing
        echo "  Testing ASM parsing..."
        if "$OUTPUT_DIR/differential_analyzer" "$output_binary" > "$OUTPUT_DIR/${filename}_${opt}.asm" 2>&1; then
            echo "    ASM parsing successful"
        else
            echo "    ASM parsing failed"
        fi
    done
done

# Generate summary
echo
echo "Differential Analysis Summary"
echo "=============================="

echo "Processed files:"
ls -1 "$OUTPUT_DIR"/*.diff 2>/dev/null | wc -l

echo
echo "Sample differential results:"
find "$OUTPUT_DIR" -name "*.diff" -exec head -5 {} \; | head -20

echo
echo "Differential extraction complete!"
echo "Results saved in: $OUTPUT_DIR"