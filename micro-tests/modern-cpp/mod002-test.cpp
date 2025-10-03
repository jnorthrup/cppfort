// mod002-test.cpp
// Modern C++ test 2
// Test #702


#include <utility>
auto func() { return std::make_pair(2, 2+1); }
int main() { auto [a,b] = func(); return a; }
