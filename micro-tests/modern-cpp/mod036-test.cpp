// mod036-test.cpp
// Modern C++ test 36
// Test #736


#include <utility>
auto func() { return std::make_pair(36, 36+1); }
int main() { auto [a,b] = func(); return a; }
