// cf090-short-circuit-ternary.cpp
// Short-circuit in ternary operator
// Test #090


int test_short_circuit_ternary(int x) {
    int counter = 0;
    int result = (x > 0) ? (counter++, 10) : (counter += 5, 20);
    return counter;
}

int main() {
    return test_short_circuit_ternary(5);
}
