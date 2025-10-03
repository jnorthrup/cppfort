// cf027-range-for-const-ref.cpp
// Range-based for with const reference
// Test #027


int test_range_for_const_ref() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (const int& x : arr) {
        sum += x;
    }
    return sum;
}

int main() {
    return test_range_for_const_ref();
}
