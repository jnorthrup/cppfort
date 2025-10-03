// cf035-nested-while.cpp
// Nested while loops
// Test #035


int test_nested_while() {
    int sum = 0;
    int i = 0;
    while (i < 5) {
        int j = 0;
        while (j < 5) {
            sum += i * j;
            j++;
        }
        i++;
    }
    return sum;
}

int main() {
    return test_nested_while();
}
