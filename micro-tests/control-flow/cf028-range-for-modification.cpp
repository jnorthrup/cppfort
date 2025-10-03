// cf028-range-for-modification.cpp
// Range-based for with modification via reference
// Test #028


void test_range_for_modify(int arr[5]) {
    for (int& x : arr) {
        x *= 2;
    }
}

int main() {
    int arr[5] = {1, 2, 3, 4, 5};
    test_range_for_modify(arr);
    return arr[0] + arr[4];
}
