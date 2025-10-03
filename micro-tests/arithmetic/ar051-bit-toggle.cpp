// ar051-bit-toggle.cpp
// Toggle bit
// Test #051


int test_bit_toggle(int value, int bit) {
    return value ^ (1 << bit);
}

int main() {
    return test_bit_toggle(0b1010, 1);
}
