// cf012-nested-ternary.cpp
// Nested ternary operators
// Test #012


int test_nested_ternary(int x) {
    return (x > 0) ? (x > 10 ? 2 : 1) : 0;
}

int main() {
    return test_nested_ternary(5);
}
