// mod004-test.cpp
// Modern C++ test 4
// Test #704


#include <utility>
auto func() { return std::make_pair(4, 4+1); }
int main() { auto [a,b] = func(); return a; }
