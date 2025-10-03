// ar018-precedence.cpp
// Operator precedence
// Test #018


int test_precedence(int a, int b, int c) {
    return a + b * c;
}

int main() {
    return test_precedence(2, 3, 4);
}
