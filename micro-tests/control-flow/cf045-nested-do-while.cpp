// cf045-nested-do-while.cpp
// Nested do-while loops
// Test #045


int test_nested_do_while() {
    int sum = 0;
    int i = 0;
    do {
        int j = 0;
        do {
            sum += i * j;
            j++;
        } while (j < 5);
        i++;
    } while (i < 5);
    return sum;
}

int main() {
    return test_nested_do_while();
}
