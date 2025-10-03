// ar042-unsigned-right-shift.cpp
// Unsigned right shift
// Test #042


unsigned int test_unsigned_shift(unsigned int x, int n) {
    return x >> n;
}

int main() {
    return test_unsigned_shift(0x80000000, 1);
}
