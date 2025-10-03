// cf015-ternary-complex-expr.cpp
// Ternary with complex expressions in both branches
// Test #015


int test_ternary_complex(int x, int y) {
    return (x > y) ? (x + y) : (x - y);
}

int main() {
    return test_ternary_complex(10, 3);
}
