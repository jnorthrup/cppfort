// mem072-reference-assignment.cpp
// Assignment through reference
// Test #152


int test_reference_assignment() {
    int x = 10;
    int& ref = x;
    ref = 20;
    return x;
}

int main() {
    return test_reference_assignment();
}
