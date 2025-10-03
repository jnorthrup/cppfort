#!/bin/bash
# Test script for stage0->IR->stage1 pipeline integration

set -e

echo "=== Testing Stage0 -> IR -> Stage1 Integration ==="
echo

# Test 1: Simple function
echo "Test 1: Simple function with no parameters"
cat > test_simple.cpp2 <<'EOF'
main: () -> int = {
    return 0;
}
EOF

./build/stage0_cli transpile test_simple.cpp2 test_simple.ir --backend ir
echo "  ✓ Generated IR from cpp2"

./build/stage1_cli transpile test_simple.ir test_simple.cpp --input-format ir
echo "  ✓ Generated C++ from IR"

# Test 2: Function with parameters
echo
echo "Test 2: Function with in parameters"
cat > test_params.cpp2 <<'EOF'
add: (in x: int, in y: int) -> int = {
    return x + y;
}

main: () -> int = {
    result: int = add(10, 20);
    return result;
}
EOF

./build/stage0_cli transpile test_params.cpp2 test_params.ir --backend ir
echo "  ✓ Generated IR from cpp2 with parameters"

./build/stage1_cli transpile test_params.ir test_params.cpp --input-format ir
echo "  ✓ Generated C++ from IR with parameters"

# Test 3: Different parameter kinds
echo
echo "Test 3: Different parameter kinds (in, out, inout)"
cat > test_param_kinds.cpp2 <<'EOF'
modify: (in x: int, out y: int, inout z: int) -> int = {
    y = x + 1;
    z = z * 2;
    return x + y + z;
}
EOF

./build/stage0_cli transpile test_param_kinds.cpp2 test_param_kinds.ir --backend ir
echo "  ✓ Generated IR from cpp2 with different parameter kinds"

./build/stage1_cli transpile test_param_kinds.ir test_param_kinds.cpp --input-format ir
echo "  ✓ Generated C++ from IR with different parameter kinds"

echo
echo "=== All Integration Tests Passed ==="
echo
echo "IR format allows stage0 and stage1 to communicate without C++ intermediate files."
echo "This enables a true multi-stage compiler pipeline."
