// mod010-test.cpp
// Modern C++ test 10
// Test #710


#include <utility>
auto func() { return std::make_pair(10, 10+1); }
int main() { auto [a,b] = func(); return a; }
