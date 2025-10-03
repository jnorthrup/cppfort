// ar045-compound-xor.cpp
// Compound bitwise XOR (^=)
// Test #045


int test_compound_xor(int x, int y) {
    x ^= y;
    return x;
}

int main() {
    return test_compound_xor(0b1111, 0b1010);
}
