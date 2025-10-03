// mem013-global-array.cpp
// Global array access
// Test #093


int global_arr[5] = {1, 2, 3, 4, 5};

int test_global_array() {
    return global_arr[2];
}

int main() {
    return test_global_array();
}
