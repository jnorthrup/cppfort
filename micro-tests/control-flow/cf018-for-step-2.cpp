// cf018-for-step-2.cpp
// For loop with step of 2
// Test #018


int test_for_step() {
    int sum = 0;
    for (int i = 0; i < 10; i += 2) {
        sum += i;
    }
    return sum;
}

int main() {
    return test_for_step();
}
