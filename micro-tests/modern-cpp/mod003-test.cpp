// mod003-test.cpp
// Modern C++ test 3
// Test #703


#include <utility>
auto func() { return std::make_pair(3, 3+1); }
int main() { auto [a,b] = func(); return a; }
