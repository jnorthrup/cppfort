// mod060-test.cpp
// Modern C++ test 60
// Test #760


#include <utility>
auto func() { return std::make_pair(60, 60+1); }
int main() { auto [a,b] = func(); return a; }
