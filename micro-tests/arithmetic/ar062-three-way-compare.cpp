// ar062-three-way-compare.cpp
// Three-way comparison (C++20)
// Test #062


#include <compare>

auto test_three_way(int a, int b) {
    return a <=> b;
}

int main() {
    auto result = test_three_way(5, 3);
    return result > 0 ? 1 : 0;
}
