// mem070-dynamic-2d-array.cpp
// Dynamic 2D array allocation
// Test #150


int test_dynamic_2d_array() {
    int** arr = new int*[3];
    for (int i = 0; i < 3; i++) {
        arr[i] = new int[3];
    }
    arr[1][1] = 42;
    int result = arr[1][1];
    for (int i = 0; i < 3; i++) {
        delete[] arr[i];
    }
    delete[] arr;
    return result;
}

int main() {
    return test_dynamic_2d_array();
}
