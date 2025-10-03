// ar048-bit-mask.cpp
// Bit masking
// Test #048


int test_bit_mask(int value) {
    return value & 0xFF;
}

int main() {
    return test_bit_mask(0x12345678);
}
