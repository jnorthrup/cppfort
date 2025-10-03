// cf036-while-decrement.cpp
// While loop with decrement
// Test #036


int test_while_decrement() {
    int sum = 0;
    int i = 10;
    while (i > 0) {
        sum += i;
        i--;
    }
    return sum;
}

int main() {
    return test_while_decrement();
}
