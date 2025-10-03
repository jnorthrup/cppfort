// mod033-test.cpp
// Modern C++ test 33
// Test #733


#include <utility>
auto func() { return std::make_pair(33, 33+1); }
int main() { auto [a,b] = func(); return a; }
