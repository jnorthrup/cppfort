// ar050-bit-clear.cpp
// Clear bit
// Test #050


int test_bit_clear(int value, int bit) {
    return value & ~(1 << bit);
}

int main() {
    return test_bit_clear(0xFF, 3);
}
