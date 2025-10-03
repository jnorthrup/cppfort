// mod047-test.cpp
// Modern C++ test 47
// Test #747


#include <utility>
auto func() { return std::make_pair(47, 47+1); }
int main() { auto [a,b] = func(); return a; }
