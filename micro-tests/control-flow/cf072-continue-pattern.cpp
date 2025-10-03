// cf072-continue-pattern.cpp
// Multiple continue conditions
// Test #072


int test_continue_pattern() {
    int sum = 0;
    for (int i = 0; i < 20; i++) {
        if (i < 5) continue;
        if (i > 15) continue;
        if (i % 2 == 0) continue;
        sum += i;
    }
    return sum;
}

int main() {
    return test_continue_pattern();
}
