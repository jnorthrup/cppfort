// mod046-test.cpp
// Modern C++ test 46
// Test #746


#include <utility>
auto func() { return std::make_pair(46, 46+1); }
int main() { auto [a,b] = func(); return a; }
