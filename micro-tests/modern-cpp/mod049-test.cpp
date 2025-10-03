// mod049-test.cpp
// Modern C++ test 49
// Test #749


#include <utility>
auto func() { return std::make_pair(49, 49+1); }
int main() { auto [a,b] = func(); return a; }
