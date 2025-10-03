// mod053-test.cpp
// Modern C++ test 53
// Test #753


#include <utility>
auto func() { return std::make_pair(53, 53+1); }
int main() { auto [a,b] = func(); return a; }
