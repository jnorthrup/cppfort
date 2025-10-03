// mod032-test.cpp
// Modern C++ test 32
// Test #732


#include <utility>
auto func() { return std::make_pair(32, 32+1); }
int main() { auto [a,b] = func(); return a; }
