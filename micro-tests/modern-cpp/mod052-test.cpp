// mod052-test.cpp
// Modern C++ test 52
// Test #752


#include <utility>
auto func() { return std::make_pair(52, 52+1); }
int main() { auto [a,b] = func(); return a; }
