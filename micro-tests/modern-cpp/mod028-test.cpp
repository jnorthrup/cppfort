// mod028-test.cpp
// Modern C++ test 28
// Test #728


#include <utility>
auto func() { return std::make_pair(28, 28+1); }
int main() { auto [a,b] = func(); return a; }
