// cf076-multiple-returns.cpp
// Function with multiple return statements
// Test #076


int test_multiple_returns(int x) {
    if (x < 0) return -1;
    if (x == 0) return 0;
    if (x < 10) return 1;
    if (x < 100) return 2;
    return 3;
}

int main() {
    return test_multiple_returns(50);
}
