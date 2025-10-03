// cf088-short-circuit-array-access.cpp
// Short-circuit in loop condition with array access
// Test #088


int test_array_short_circuit(int* arr, int size) {
    int sum = 0;
    for (int i = 0; i < size && arr && arr[i] != 0; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    int arr[5] = {1, 2, 0, 4, 5};
    return test_array_short_circuit(arr, 5);
}
