// cf083-short-circuit-null-check.cpp
// Short-circuit null pointer check
// Test #083


int test_null_check(int* ptr) {
    if (ptr && *ptr > 0) {
        return *ptr;
    }
    return 0;
}

int main() {
    int val = 42;
    return test_null_check(&val);
}
