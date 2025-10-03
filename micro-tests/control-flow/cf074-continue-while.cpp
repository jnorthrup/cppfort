// cf074-continue-while.cpp
// Continue in while loop
// Test #074


int test_continue_while() {
    int sum = 0;
    int i = 0;
    while (i < 20) {
        i++;
        if (i % 3 == 0) continue;
        sum += i;
    }
    return sum;
}

int main() {
    return test_continue_while();
}
