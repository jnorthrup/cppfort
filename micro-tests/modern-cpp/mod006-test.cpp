// mod006-test.cpp
// Modern C++ test 6
// Test #706


#include <utility>
auto func() { return std::make_pair(6, 6+1); }
int main() { auto [a,b] = func(); return a; }
