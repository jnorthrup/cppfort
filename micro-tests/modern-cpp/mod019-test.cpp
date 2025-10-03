// mod019-test.cpp
// Modern C++ test 19
// Test #719


#include <utility>
auto func() { return std::make_pair(19, 19+1); }
int main() { auto [a,b] = func(); return a; }
