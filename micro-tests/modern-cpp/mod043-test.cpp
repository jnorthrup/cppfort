// mod043-test.cpp
// Modern C++ test 43
// Test #743


#include <utility>
auto func() { return std::make_pair(43, 43+1); }
int main() { auto [a,b] = func(); return a; }
