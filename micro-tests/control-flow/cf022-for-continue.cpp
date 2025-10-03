// cf022-for-continue.cpp
// For loop with continue statement
// Test #022


int test_for_continue() {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        if (i % 2 == 0) continue;
        sum += i;
    }
    return sum;
}

int main() {
    return test_for_continue();
}
