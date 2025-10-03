// mod031-test.cpp
// Modern C++ test 31
// Test #731


#include <utility>
auto func() { return std::make_pair(31, 31+1); }
int main() { auto [a,b] = func(); return a; }
