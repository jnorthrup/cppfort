// cf017-for-decrement.cpp
// For loop with decrement
// Test #017


int test_for_decrement() {
    int sum = 0;
    for (int i = 10; i > 0; i--) {
        sum += i;
    }
    return sum;
}

int main() {
    return test_for_decrement();
}
