// mem029-pointer-const.cpp
// Const pointer
// Test #109


int test_pointer_const() {
    int x = 42, y = 10;
    int* const ptr = &x;
    // ptr = &y;  // Error: can't reassign
    return *ptr;
}

int main() {
    return test_pointer_const();
}
