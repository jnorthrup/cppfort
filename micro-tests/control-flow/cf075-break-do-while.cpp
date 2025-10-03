// cf075-break-do-while.cpp
// Break in do-while loop
// Test #075


int test_break_do_while() {
    int sum = 0;
    int i = 0;
    do {
        sum += i;
        i++;
        if (sum > 20) break;
    } while (i < 100);
    return sum;
}

int main() {
    return test_break_do_while();
}
