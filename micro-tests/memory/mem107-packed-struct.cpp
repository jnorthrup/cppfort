// mem107-packed-struct.cpp
// Packed struct (no padding)
// Test #187


struct __attribute__((packed)) Packed {
    char c;
    int i;
    char c2;
};

int test_packed_struct() {
    return sizeof(Packed);
}

int main() {
    return test_packed_struct();
}
