// cf082-short-circuit-or.cpp
// Short-circuit OR evaluation
// Test #082


bool expensive_check(int& counter) {
    counter++;
    return false;
}

int test_short_circuit_or() {
    int counter = 0;
    bool result = (true || expensive_check(counter));
    return counter;  // Should be 0 (expensive_check not called)
}

int main() {
    return test_short_circuit_or();
}
