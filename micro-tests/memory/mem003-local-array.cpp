// mem003-local-array.cpp
// Local array on stack
// Test #083


int test_local_array() {
    int arr[5] = {1, 2, 3, 4, 5};
    return arr[2];
}

int main() {
    return test_local_array();
}
