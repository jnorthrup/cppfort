// cf084-short-circuit-bounds-check.cpp
// Short-circuit bounds checking
// Test #084


int test_bounds_check(int* arr, int size, int index) {
    if (index >= 0 && index < size && arr[index] > 0) {
        return arr[index];
    }
    return -1;
}

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    return test_bounds_check(arr, 5, 2);
}
