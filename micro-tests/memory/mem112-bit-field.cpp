// mem112-bit-field.cpp
// Bit fields
// Test #192


struct BitField {
    unsigned int a : 3;
    unsigned int b : 5;
    unsigned int c : 8;
};

int test_bit_field() {
    BitField bf;
    bf.a = 7;
    bf.b = 31;
    bf.c = 255;
    return bf.a + bf.b;
}

int main() {
    return test_bit_field();
}
