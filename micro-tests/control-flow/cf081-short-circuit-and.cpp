// cf081-short-circuit-and.cpp
// Short-circuit AND evaluation
// Test #081


bool expensive_check(int& counter) {
    counter++;
    return true;
}

int test_short_circuit_and() {
    int counter = 0;
    bool result = (false && expensive_check(counter));
    return counter;  // Should be 0 (expensive_check not called)
}

int main() {
    return test_short_circuit_and();
}
