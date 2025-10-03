// mod038-test.cpp
// Modern C++ test 38
// Test #738


#include <utility>
auto func() { return std::make_pair(38, 38+1); }
int main() { auto [a,b] = func(); return a; }
