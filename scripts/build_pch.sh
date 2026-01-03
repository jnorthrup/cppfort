#!/bin/bash
# Build precompiled header for cppfort-generated code
# Run once, then use -include-pch for all compilations

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INCLUDE_DIR="${SCRIPT_DIR}/../include"
BUILD_DIR="${SCRIPT_DIR}/../build"

mkdir -p "$BUILD_DIR/pch"

CXX="${CXX:-clang++}"
CXXFLAGS="${CXXFLAGS:--std=c++20 -O2}"

echo "Building precompiled header with $CXX..."

if [[ "$CXX" == *"clang"* ]]; then
    # Clang: produces .pch file
    $CXX $CXXFLAGS -x c++-header \
        -I "$INCLUDE_DIR" \
        "$INCLUDE_DIR/cpp2_pch.h" \
        -o "$BUILD_DIR/pch/cpp2_pch.h.pch"
    
    echo "Created: $BUILD_DIR/pch/cpp2_pch.h.pch"
    echo ""
    echo "Usage:"
    echo "  $CXX $CXXFLAGS -include-pch $BUILD_DIR/pch/cpp2_pch.h.pch mycode.cpp -o mycode"
    
elif [[ "$CXX" == *"g++"* ]]; then
    # GCC: produces .gch file (must be in same dir as header or include path)
    $CXX $CXXFLAGS -x c++-header \
        -I "$INCLUDE_DIR" \
        "$INCLUDE_DIR/cpp2_pch.h" \
        -o "$INCLUDE_DIR/cpp2_pch.h.gch"
    
    echo "Created: $INCLUDE_DIR/cpp2_pch.h.gch"
    echo ""
    echo "Usage:"
    echo "  $CXX $CXXFLAGS -I $INCLUDE_DIR -include cpp2_pch.h mycode.cpp -o mycode"
    
else
    echo "Unknown compiler: $CXX"
    exit 1
fi

echo ""
echo "Speedup: ~3-5x faster compilation for generated Cpp2 code"
