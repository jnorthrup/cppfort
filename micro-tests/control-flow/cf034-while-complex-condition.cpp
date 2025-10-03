// cf034-while-complex-condition.cpp
// While loop with complex condition
// Test #034


int test_while_complex() {
    int sum = 0;
    int i = 0;
    while (i < 10 && sum < 20) {
        sum += i;
        i++;
    }
    return sum;
}

int main() {
    return test_while_complex();
}
