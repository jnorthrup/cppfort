// mod014-test.cpp
// Modern C++ test 14
// Test #714


#include <utility>
auto func() { return std::make_pair(14, 14+1); }
int main() { auto [a,b] = func(); return a; }
