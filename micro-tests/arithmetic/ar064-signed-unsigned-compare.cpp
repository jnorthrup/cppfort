// ar064-signed-unsigned-compare.cpp
// Mixed signed/unsigned comparison
// Test #064


int test_mixed_sign_compare(int a, unsigned int b) {
    return a < (int)b;
}

int main() {
    return test_mixed_sign_compare(-5, 10);
}
