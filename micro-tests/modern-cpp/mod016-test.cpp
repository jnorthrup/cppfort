// mod016-test.cpp
// Modern C++ test 16
// Test #716


#include <utility>
auto func() { return std::make_pair(16, 16+1); }
int main() { auto [a,b] = func(); return a; }
