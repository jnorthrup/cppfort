// cf032-while-break.cpp
// While loop with break
// Test #032


int test_while_break() {
    int sum = 0;
    int i = 0;
    while (true) {
        if (i >= 10) break;
        sum += i;
        i++;
    }
    return sum;
}

int main() {
    return test_while_break();
}
