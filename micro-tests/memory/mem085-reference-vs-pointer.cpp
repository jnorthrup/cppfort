// mem085-reference-vs-pointer.cpp
// Reference vs pointer comparison
// Test #165


int test_ref_vs_ptr() {
    int x = 42;
    int& ref = x;
    int* ptr = &x;
    return (ref == *ptr) ? 1 : 0;
}

int main() {
    return test_ref_vs_ptr();
}
