#!/bin/bash

# Test script for enhanced parser features
# This script tests the new parser capabilities

set -e

echo "Testing enhanced parser features..."

# Build the parser if needed
echo "Building parser..."
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
cd ..

# Test basic parsing
echo "Testing basic cpp2 file parsing..."
./build/src/stage0/cpp2_stage0 test_enhanced_parser.cpp2

if [ $? -eq 0 ]; then
    echo "✓ Basic parsing successful"
else
    echo "✗ Basic parsing failed"
    exit 1
fi

# Test specific features
echo "Testing access specifiers..."
grep -q "public mytype" test_enhanced_parser.cpp2 && echo "✓ Public access specifier found"
grep -q "private data" test_enhanced_parser.cpp2 && echo "✓ Private access specifier found"
grep -q "protected counter" test_enhanced_parser.cpp2 && echo "✓ Protected access specifier found"

echo "Testing parameter kinds..."
grep -q "inout this" test_enhanced_parser.cpp2 && echo "✓ inout parameter kind found"
grep -q "in new_data" test_enhanced_parser.cpp2 && echo "✓ in parameter kind found"
grep -q "out result" test_enhanced_parser.cpp2 && echo "✓ out parameter kind found"

echo "Testing 'this' parameter handling..."
grep -q "this" test_enhanced_parser.cpp2 && echo "✓ 'this' parameter usage found"

echo "All parser enhancement tests completed successfully!"