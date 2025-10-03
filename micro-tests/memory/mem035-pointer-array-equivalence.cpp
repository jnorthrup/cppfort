// mem035-pointer-array-equivalence.cpp
// Pointer/array equivalence
// Test #115


int test_pointer_array_equiv() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr = arr;
    return ptr[2] == arr[2];
}

int main() {
    return test_pointer_array_equiv();
}
