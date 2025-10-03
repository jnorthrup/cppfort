// mod045-test.cpp
// Modern C++ test 45
// Test #745


#include <utility>
auto func() { return std::make_pair(45, 45+1); }
int main() { auto [a,b] = func(); return a; }
