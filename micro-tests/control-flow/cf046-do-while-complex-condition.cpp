// cf046-do-while-complex-condition.cpp
// Do-while with complex condition
// Test #046


int test_do_while_complex() {
    int sum = 0;
    int i = 0;
    do {
        sum += i;
        i++;
    } while (i < 10 && sum < 20);
    return sum;
}

int main() {
    return test_do_while_complex();
}
