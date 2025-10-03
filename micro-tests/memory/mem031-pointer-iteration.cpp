// mem031-pointer-iteration.cpp
// Array iteration with pointers
// Test #111


int test_pointer_iteration() {
    int arr[5] = {1, 2, 3, 4, 5};
    int sum = 0;
    for (int* ptr = arr; ptr < arr + 5; ptr++) {
        sum += *ptr;
    }
    return sum;
}

int main() {
    return test_pointer_iteration();
}
