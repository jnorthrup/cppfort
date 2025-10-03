// mod059-test.cpp
// Modern C++ test 59
// Test #759


#include <utility>
auto func() { return std::make_pair(59, 59+1); }
int main() { auto [a,b] = func(); return a; }
