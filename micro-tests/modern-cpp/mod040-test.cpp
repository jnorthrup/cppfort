// mod040-test.cpp
// Modern C++ test 40
// Test #740


#include <utility>
auto func() { return std::make_pair(40, 40+1); }
int main() { auto [a,b] = func(); return a; }
