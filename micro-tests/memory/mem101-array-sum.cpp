// mem101-array-sum.cpp
// Array sum
// Test #181


int test_array_sum() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    return test_array_sum();
}
