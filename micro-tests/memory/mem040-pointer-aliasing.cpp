// mem040-pointer-aliasing.cpp
// Pointer aliasing
// Test #120


int test_pointer_aliasing() {
    int x = 10;
    int* ptr1 = &x;
    int* ptr2 = &x;
    *ptr1 = 20;
    return *ptr2;
}

int main() {
    return test_pointer_aliasing();
}
