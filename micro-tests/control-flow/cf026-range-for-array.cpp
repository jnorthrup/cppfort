// cf026-range-for-array.cpp
// Range-based for loop over array
// Test #026


int test_range_for() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int x : arr) {
        sum += x;
    }
    return sum;
}

int main() {
    return test_range_for();
}
