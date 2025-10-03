// ar070-sign-function.cpp
// Sign function
// Test #070


int test_sign(int x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

int main() {
    return test_sign(-5);
}
