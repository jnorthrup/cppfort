// cf085-short-circuit-complex.cpp
// Complex short-circuit evaluation
// Test #085


bool check1(int& c) { c += 1; return true; }
bool check2(int& c) { c += 10; return false; }
bool check3(int& c) { c += 100; return true; }

int test_complex_short_circuit() {
    int counter = 0;
    bool result = check1(counter) && check2(counter) && check3(counter);
    return counter;  // Should be 11 (check3 not called)
}

int main() {
    return test_complex_short_circuit();
}
