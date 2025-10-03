// mod056-test.cpp
// Modern C++ test 56
// Test #756


#include <utility>
auto func() { return std::make_pair(56, 56+1); }
int main() { auto [a,b] = func(); return a; }
