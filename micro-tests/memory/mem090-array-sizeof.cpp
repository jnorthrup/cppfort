// mem090-array-sizeof.cpp
// Array size using sizeof
// Test #170


int test_array_sizeof() {
    int arr[5] = {1, 2, 3, 4, 5};
    return sizeof(arr) / sizeof(arr[0]);
}

int main() {
    return test_array_sizeof();
}
