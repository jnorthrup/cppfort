// cf047-do-while-decrement.cpp
// Do-while with decrement
// Test #047


int test_do_while_decrement() {
    int sum = 0;
    int i = 10;
    do {
        sum += i;
        i--;
    } while (i > 0);
    return sum;
}

int main() {
    return test_do_while_decrement();
}
