// mem025-pointer-to-array.cpp
// Pointer to array
// Test #105


int test_pointer_to_array() {
    int arr[3] = {1, 2, 3};
    int (*ptr)[3] = &arr;
    return (*ptr)[1];
}

int main() {
    return test_pointer_to_array();
}
