// mod020-test.cpp
// Modern C++ test 20
// Test #720


#include <utility>
auto func() { return std::make_pair(20, 20+1); }
int main() { auto [a,b] = func(); return a; }
