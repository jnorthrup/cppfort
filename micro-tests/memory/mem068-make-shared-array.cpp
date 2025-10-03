// mem068-make-shared-array.cpp
// Make shared for arrays (C++20)
// Test #148


#include <memory>

int test_make_shared_array() {
    std::shared_ptr<int[]> arr = std::make_shared<int[]>(5);
    arr[2] = 42;
    return arr[2];
}

int main() {
    return test_make_shared_array();
}
