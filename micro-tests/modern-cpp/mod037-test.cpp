// mod037-test.cpp
// Modern C++ test 37
// Test #737


#include <utility>
auto func() { return std::make_pair(37, 37+1); }
int main() { auto [a,b] = func(); return a; }
