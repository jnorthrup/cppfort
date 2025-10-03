// ar049-bit-set.cpp
// Set bit
// Test #049


int test_bit_set(int value, int bit) {
    return value | (1 << bit);
}

int main() {
    return test_bit_set(0, 3);
}
