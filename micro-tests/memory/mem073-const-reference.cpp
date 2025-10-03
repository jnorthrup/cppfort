// mem073-const-reference.cpp
// Const reference
// Test #153


int test_const_reference() {
    int x = 42;
    const int& ref = x;
    return ref;
}

int main() {
    return test_const_reference();
}
