// cf016-simple-for.cpp
// Simple for loop with increment
// Test #016


int test_simple_for() {
    int sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += i;
    }
    return sum;
}

int main() {
    return test_simple_for();
}
