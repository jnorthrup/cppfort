// cf014-ternary-function-calls.cpp
// Ternary operator with function calls
// Test #014


int positive_func() { return 1; }
int negative_func() { return -1; }

int test_ternary_calls(int x) {
    return (x > 0) ? positive_func() : negative_func();
}

int main() {
    return test_ternary_calls(5);
}
