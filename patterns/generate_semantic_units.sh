#!/bin/bash
# Generate C++ code from semantic_units.td using LLVM tblgen

TBLGEN="llvm-tblgen"
TD_FILE="semantic_units.td"
OUTPUT_DIR="../src/stage0"

# Check if tblgen is available
if ! command -v $TBLGEN &> /dev/null; then
    echo "ERROR: llvm-tblgen not found in PATH"
    echo "Install LLVM development tools or add to PATH"
    exit 1
fi

# Generate C++ declarations
echo "Generating semantic unit declarations..."
$TBLGEN -gen-op-decls $TD_FILE -o ${OUTPUT_DIR}/semantic_units_gen.h

# Generate C++ definitions
echo "Generating semantic unit definitions..."
$TBLGEN -gen-op-defs $TD_FILE -o ${OUTPUT_DIR}/semantic_units_gen.cpp

echo "Generated files:"
echo "  ${OUTPUT_DIR}/semantic_units_gen.h"
echo "  ${OUTPUT_DIR}/semantic_units_gen.cpp"
