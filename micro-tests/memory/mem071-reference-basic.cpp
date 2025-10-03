// mem071-reference-basic.cpp
// Basic reference
// Test #151


int test_reference() {
    int x = 42;
    int& ref = x;
    return ref;
}

int main() {
    return test_reference();
}
