// cf089-short-circuit-function-calls.cpp
// Short-circuit with function calls
// Test #089


int func1() { return 0; }
int func2() { return 1; }
int func3() { return 2; }

int test_function_short_circuit() {
    return func1() || func2() || func3();
}

int main() {
    return test_function_short_circuit();
}
