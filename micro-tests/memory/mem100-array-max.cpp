// mem100-array-max.cpp
// Find maximum in array
// Test #180


int test_array_max() {
    int arr[5] = {3, 7, 2, 9, 1};
    int max = arr[0];
    for (int i = 1; i < 5; i++) {
        if (arr[i] > max) max = arr[i];
    }
    return max;
}

int main() {
    return test_array_max();
}
