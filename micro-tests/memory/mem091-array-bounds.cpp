// mem091-array-bounds.cpp
// Array bounds (last valid element)
// Test #171


int test_array_bounds() {
    int arr[5] = {1, 2, 3, 4, 5};
    // Accessing arr[5] is out of bounds!
    return arr[4];
}

int main() {
    return test_array_bounds();
}
