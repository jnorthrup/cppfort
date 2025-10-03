// mem016-pointer-basic.cpp
// Basic pointer dereference
// Test #096


int test_pointer_basic() {
    int x = 42;
    int* ptr = &x;
    return *ptr;
}

int main() {
    return test_pointer_basic();
}
