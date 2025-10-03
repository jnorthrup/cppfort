// mem026-array-of-pointers.cpp
// Array of pointers
// Test #106


int test_array_of_pointers() {
    int a = 1, b = 2, c = 3;
    int* arr[3] = {&a, &b, &c};
    return *arr[1];
}

int main() {
    return test_array_of_pointers();
}
