// mem103-std-array.cpp
// std::array (C++11)
// Test #183


#include <array>

int test_std_array() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    return arr[2];
}

int main() {
    return test_std_array();
}
