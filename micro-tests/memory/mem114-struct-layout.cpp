// mem114-struct-layout.cpp
// Struct memory layout
// Test #194


struct Layout {
    char a;
    short b;
    int c;
    long long d;
};

int test_struct_layout() {
    return sizeof(Layout);
}

int main() {
    return test_struct_layout();
}
