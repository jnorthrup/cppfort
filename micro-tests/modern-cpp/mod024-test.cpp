// mod024-test.cpp
// Modern C++ test 24
// Test #724


#include <utility>
auto func() { return std::make_pair(24, 24+1); }
int main() { auto [a,b] = func(); return a; }
