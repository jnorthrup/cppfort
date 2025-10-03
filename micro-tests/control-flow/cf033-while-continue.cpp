// cf033-while-continue.cpp
// While loop with continue
// Test #033


int test_while_continue() {
    int sum = 0;
    int i = 0;
    while (i < 10) {
        i++;
        if (i % 2 == 0) continue;
        sum += i;
    }
    return sum;
}

int main() {
    return test_while_continue();
}
