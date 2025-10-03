// cf029-for-infinite-with-break.cpp
// Infinite for loop with break
// Test #029


int test_infinite_for() {
    int sum = 0;
    for (;;) {
        sum++;
        if (sum >= 10) break;
    }
    return sum;
}

int main() {
    return test_infinite_for();
}
