// mod051-test.cpp
// Modern C++ test 51
// Test #751


#include <utility>
auto func() { return std::make_pair(51, 51+1); }
int main() { auto [a,b] = func(); return a; }
