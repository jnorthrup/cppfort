// ar052-bit-test.cpp
// Test bit
// Test #052


int test_bit_test(int value, int bit) {
    return (value & (1 << bit)) != 0;
}

int main() {
    return test_bit_test(0b1000, 3);
}
