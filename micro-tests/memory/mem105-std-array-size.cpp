// mem105-std-array-size.cpp
// std::array size method
// Test #185


#include <array>

int test_std_array_size() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    return arr.size();
}

int main() {
    return test_std_array_size();
}
