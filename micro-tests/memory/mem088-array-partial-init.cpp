// mem088-array-partial-init.cpp
// Array partial initialization
// Test #168


int test_array_partial_init() {
    int arr[5] = {1, 2};
    return arr[0] + arr[4];
}

int main() {
    return test_array_partial_init();
}
