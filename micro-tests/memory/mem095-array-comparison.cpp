// mem095-array-comparison.cpp
// Array comparison
// Test #175


int test_array_comparison() {
    int arr1[3] = {1, 2, 3};
    int arr2[3] = {1, 2, 3};
    for (int i = 0; i < 3; i++) {
        if (arr1[i] != arr2[i]) return 0;
    }
    return 1;
}

int main() {
    return test_array_comparison();
}
