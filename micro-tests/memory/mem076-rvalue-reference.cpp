// mem076-rvalue-reference.cpp
// Rvalue reference
// Test #156


int test_rvalue_reference(int&& x) {
    return x;
}

int main() {
    return test_rvalue_reference(42);
}
