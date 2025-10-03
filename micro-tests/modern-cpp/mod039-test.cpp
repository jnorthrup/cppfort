// mod039-test.cpp
// Modern C++ test 39
// Test #739


#include <utility>
auto func() { return std::make_pair(39, 39+1); }
int main() { auto [a,b] = func(); return a; }
