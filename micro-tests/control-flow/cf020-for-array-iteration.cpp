// cf020-for-array-iteration.cpp
// For loop iterating over array
// Test #020


int test_for_array() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int i = 0; i < 5; i++) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    return test_for_array();
}
