// cf086-short-circuit-side-effects.cpp
// Short-circuit with side effects
// Test #086


int test_side_effects(int x) {
    int result = 0;
    (x > 0) && (result = 10, true);
    return result;
}

int main() {
    return test_side_effects(5);
}
