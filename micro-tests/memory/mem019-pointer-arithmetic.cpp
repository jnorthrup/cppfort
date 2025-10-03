// mem019-pointer-arithmetic.cpp
// Pointer arithmetic
// Test #099


int test_pointer_arithmetic() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr = arr;
    ptr += 2;
    return *ptr;
}

int main() {
    return test_pointer_arithmetic();
}
