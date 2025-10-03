// ar020-complex-expr.cpp
// Complex arithmetic expression
// Test #020


int test_complex(int a, int b, int c, int d) {
    return (a + b) * (c - d) / 2;
}

int main() {
    return test_complex(10, 5, 20, 4);
}
