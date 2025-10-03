// ar054-bit-count.cpp
// Count set bits
// Test #054


int test_bit_count(unsigned int x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

int main() {
    return test_bit_count(0b10101010);
}
