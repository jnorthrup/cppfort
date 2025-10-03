// mem087-array-zero-init.cpp
// Array zero initialization
// Test #167


int test_array_zero_init() {
    int arr[5] = {0};
    return arr[0] + arr[4];
}

int main() {
    return test_array_zero_init();
}
