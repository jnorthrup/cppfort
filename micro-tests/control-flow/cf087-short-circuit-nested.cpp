// cf087-short-circuit-nested.cpp
// Nested short-circuit conditions
// Test #087


int test_nested_short_circuit(int x, int y, int z) {
    if ((x > 0 && y > 0) || (z > 0)) {
        return 1;
    }
    return 0;
}

int main() {
    return test_nested_short_circuit(0, 0, 5);
}
