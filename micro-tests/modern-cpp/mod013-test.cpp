// mod013-test.cpp
// Modern C++ test 13
// Test #713


#include <utility>
auto func() { return std::make_pair(13, 13+1); }
int main() { auto [a,b] = func(); return a; }
