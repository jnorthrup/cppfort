// mem021-pointer-decrement.cpp
// Pointer decrement
// Test #101


int test_pointer_decrement() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr = arr + 2;
    return *(--ptr);
}

int main() {
    return test_pointer_decrement();
}
