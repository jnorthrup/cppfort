// mem060-unique-ptr-array.cpp
// Unique pointer to array
// Test #140


#include <memory>

int test_unique_ptr_array() {
    std::unique_ptr<int[]> arr = std::make_unique<int[]>(5);
    arr[2] = 42;
    return arr[2];
}

int main() {
    return test_unique_ptr_array();
}
