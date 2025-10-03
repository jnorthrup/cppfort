// mod027-test.cpp
// Modern C++ test 27
// Test #727


#include <utility>
auto func() { return std::make_pair(27, 27+1); }
int main() { auto [a,b] = func(); return a; }
