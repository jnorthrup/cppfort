// cf025-for-complex-condition.cpp
// For loop with complex condition
// Test #025


int test_for_complex_cond() {
    int sum = 0;
    for (int i = 0; i < 10 && sum < 20; i++) {
        sum += i;
    }
    return sum;
}

int main() {
    return test_for_complex_cond();
}
