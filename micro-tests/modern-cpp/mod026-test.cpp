// mod026-test.cpp
// Modern C++ test 26
// Test #726


#include <utility>
auto func() { return std::make_pair(26, 26+1); }
int main() { auto [a,b] = func(); return a; }
