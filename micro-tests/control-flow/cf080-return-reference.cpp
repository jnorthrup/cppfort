// cf080-return-reference.cpp
// Return by reference
// Test #080


int global = 42;

int& test_return_reference() {
    return global;
}

int main() {
    int& ref = test_return_reference();
    ref = 100;
    return global;
}
