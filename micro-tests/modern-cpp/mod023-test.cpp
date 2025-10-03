// mod023-test.cpp
// Modern C++ test 23
// Test #723


#include <utility>
auto func() { return std::make_pair(23, 23+1); }
int main() { auto [a,b] = func(); return a; }
