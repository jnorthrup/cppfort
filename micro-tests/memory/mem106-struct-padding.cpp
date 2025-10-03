// mem106-struct-padding.cpp
// Struct padding
// Test #186


struct Padded {
    char c;
    int i;
    char c2;
};

int test_struct_padding() {
    return sizeof(Padded);
}

int main() {
    return test_struct_padding();
}
