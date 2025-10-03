// cf024-for-multiple-init.cpp
// For loop with multiple initializers and increments
// Test #024


int test_for_multi_init() {
    int sum = 0;
    for (int i = 0, j = 10; i < j; i++, j--) {
        sum += i + j;
    }
    return sum;
}

int main() {
    return test_for_multi_init();
}
