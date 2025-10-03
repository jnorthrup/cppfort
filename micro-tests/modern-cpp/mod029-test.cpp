// mod029-test.cpp
// Modern C++ test 29
// Test #729


#include <utility>
auto func() { return std::make_pair(29, 29+1); }
int main() { auto [a,b] = func(); return a; }
