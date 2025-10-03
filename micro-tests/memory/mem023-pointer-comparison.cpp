// mem023-pointer-comparison.cpp
// Pointer comparison
// Test #103


int test_pointer_comparison() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr1 = arr;
    int* ptr2 = arr + 3;
    return ptr1 < ptr2;
}

int main() {
    return test_pointer_comparison();
}
