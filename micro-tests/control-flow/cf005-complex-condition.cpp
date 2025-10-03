// cf005-complex-condition.cpp
// If with complex logical AND condition
// Test #005


int test_complex_condition(int x, int y) {
    if (x > 0 && y < 10) {
        return 1;
    }
    return 0;
}

int main() {
    return test_complex_condition(5, 3);
}
