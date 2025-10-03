// ar079-bitwise-vs-logical.cpp
// Bitwise vs logical operators
// Test #079


int test_bitwise_vs_logical(int a, int b) {
    return (a & b) != (a && b);
}

int main() {
    return test_bitwise_vs_logical(2, 3);
}
