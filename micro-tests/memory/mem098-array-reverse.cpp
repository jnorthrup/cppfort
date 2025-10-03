// mem098-array-reverse.cpp
// Array reverse in place
// Test #178


int test_array_reverse() {
    int arr[5] = {1, 2, 3, 4, 5};
    for (int i = 0; i < 5/2; i++) {
        int temp = arr[i];
        arr[i] = arr[4-i];
        arr[4-i] = temp;
    }
    return arr[0];
}

int main() {
    return test_array_reverse();
}
