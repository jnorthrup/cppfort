// mem020-pointer-increment.cpp
// Pointer increment
// Test #100


int test_pointer_increment() {
    int arr[5] = {1, 2, 3, 4, 5};
    int* ptr = arr;
    return *(++ptr);
}

int main() {
    return test_pointer_increment();
}
