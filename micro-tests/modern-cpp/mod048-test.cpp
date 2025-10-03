// mod048-test.cpp
// Modern C++ test 48
// Test #748


#include <utility>
auto func() { return std::make_pair(48, 48+1); }
int main() { auto [a,b] = func(); return a; }
