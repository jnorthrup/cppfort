// mem092-array-iteration.cpp
// Array iteration
// Test #172


int test_array_iteration() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    return test_array_iteration();
}
