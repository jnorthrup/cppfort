// mem079-reference-return.cpp
// Return reference from function
// Test #159


int& get_reference() {
    static int x = 42;
    return x;
}

int main() {
    int& ref = get_reference();
    return ref;
}
