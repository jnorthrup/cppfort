// mem009-array-decay.cpp
// Array decay to pointer
// Test #089


int test_array_decay(int arr[]) {
    return arr[0];
}

int main() {
    int arr[3] = {1, 2, 3};
    return test_array_decay(arr);
}
