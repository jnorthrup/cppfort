// cf009-if-side-effect.cpp
// If statement with side effects on reference parameter
// Test #009


void test_if_side_effect(int x, int& result) {
    if (x > 0) {
        result = x * 2;
    }
}

int main() {
    int r = 0;
    test_if_side_effect(5, r);
    return r;
}
