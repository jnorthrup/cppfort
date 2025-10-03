// mem084-reference-chaining.cpp
// Reference chaining
// Test #164


int test_reference_chaining() {
    int x = 42;
    int& ref1 = x;
    int& ref2 = ref1;
    return ref2;
}

int main() {
    return test_reference_chaining();
}
