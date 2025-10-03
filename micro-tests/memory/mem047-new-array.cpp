// mem047-new-array.cpp
// New and delete array
// Test #127


int test_new_array() {
    int* arr = new int[5]{1, 2, 3, 4, 5};
    int result = arr[2];
    delete[] arr;
    return result;
}

int main() {
    return test_new_array();
}
