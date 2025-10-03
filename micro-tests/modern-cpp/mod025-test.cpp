// mod025-test.cpp
// Modern C++ test 25
// Test #725


#include <utility>
auto func() { return std::make_pair(25, 25+1); }
int main() { auto [a,b] = func(); return a; }
