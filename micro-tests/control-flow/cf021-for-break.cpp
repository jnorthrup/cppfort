// cf021-for-break.cpp
// For loop with break statement
// Test #021


int test_for_break() {
    int sum = 0;
    for (int i = 0; i < 100; i++) {
        if (i >= 10) break;
        sum += i;
    }
    return sum;
}

int main() {
    return test_for_break();
}
