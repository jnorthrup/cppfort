// mem022-pointer-difference.cpp
// Pointer difference
// Test #102


int test_pointer_difference() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr1 = arr;
    int* ptr2 = arr + 3;
    return ptr2 - ptr1;
}

int main() {
    return test_pointer_difference();
}
