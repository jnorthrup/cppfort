// mod009-test.cpp
// Modern C++ test 9
// Test #709


#include <utility>
auto func() { return std::make_pair(9, 9+1); }
int main() { auto [a,b] = func(); return a; }
