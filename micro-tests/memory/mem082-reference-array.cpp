// mem082-reference-array.cpp
// Reference to array
// Test #162


int test_reference_array() {
    int arr[3] = {1, 2, 3};
    int (&ref)[3] = arr;
    return ref[1];
}

int main() {
    return test_reference_array();
}
