// mem034-pointer-offset.cpp
// Pointer offset access
// Test #114


int test_pointer_offset() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr = arr;
    return *(ptr + 2);
}

int main() {
    return test_pointer_offset();
}
