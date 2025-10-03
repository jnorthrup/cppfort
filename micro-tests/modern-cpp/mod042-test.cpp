// mod042-test.cpp
// Modern C++ test 42
// Test #742


#include <utility>
auto func() { return std::make_pair(42, 42+1); }
int main() { auto [a,b] = func(); return a; }
