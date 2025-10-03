// ar053-swap-xor.cpp
// Swap using XOR
// Test #053


void test_swap_xor(int& a, int& b) {
    a ^= b;
    b ^= a;
    a ^= b;
}

int main() {
    int x = 5, y = 10;
    test_swap_xor(x, y);
    return x;
}
