// test_bitwise.cpp - Test bitwise operations in Cpp2
auto main() -> int {
    x: int = 1;
    y: int = 2;
    z: int = x & y;      // bitwise AND
    w: int = x | y;      // bitwise OR
    v: int = x ^ y;      // bitwise XOR
    u: int = x << 1;     // shift left
    t: int = y >> 1;     // shift right
    return z + w + v + u + t;
}