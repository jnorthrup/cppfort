// cf030-for-backward.cpp
// For loop iterating backward
// Test #030


int test_for_backward() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int i = 4; i >= 0; i--) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    return test_for_backward();
}
