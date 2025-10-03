// mod015-test.cpp
// Modern C++ test 15
// Test #715


#include <utility>
auto func() { return std::make_pair(15, 15+1); }
int main() { auto [a,b] = func(); return a; }
