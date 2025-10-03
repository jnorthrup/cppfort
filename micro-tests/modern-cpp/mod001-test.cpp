// mod001-test.cpp
// Modern C++ test 1
// Test #701


#include <utility>
auto func() { return std::make_pair(1, 1+1); }
int main() { auto [a,b] = func(); return a; }
