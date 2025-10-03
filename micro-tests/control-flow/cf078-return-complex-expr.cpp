// cf078-return-complex-expr.cpp
// Return with complex expression
// Test #078


int compute(int x, int y) {
    return (x + y) * (x - y) + x * y;
}

int main() {
    return compute(5, 3);
}
