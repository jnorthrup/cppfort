// mem104-std-array-at.cpp
// std::array with bounds checking
// Test #184


#include <array>

int test_std_array_at() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    return arr.at(2);
}

int main() {
    return test_std_array_at();
}
