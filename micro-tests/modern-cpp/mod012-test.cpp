// mod012-test.cpp
// Modern C++ test 12
// Test #712


#include <utility>
auto func() { return std::make_pair(12, 12+1); }
int main() { auto [a,b] = func(); return a; }
