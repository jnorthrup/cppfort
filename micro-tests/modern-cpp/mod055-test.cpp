// mod055-test.cpp
// Modern C++ test 55
// Test #755


#include <utility>
auto func() { return std::make_pair(55, 55+1); }
int main() { auto [a,b] = func(); return a; }
