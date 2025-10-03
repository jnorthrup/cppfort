// mod011-test.cpp
// Modern C++ test 11
// Test #711


#include <utility>
auto func() { return std::make_pair(11, 11+1); }
int main() { auto [a,b] = func(); return a; }
