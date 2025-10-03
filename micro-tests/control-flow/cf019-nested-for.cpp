// cf019-nested-for.cpp
// Nested for loops
// Test #019


int test_nested_for() {
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            sum += i * j;
        }
    }
    return sum;
}

int main() {
    return test_nested_for();
}
