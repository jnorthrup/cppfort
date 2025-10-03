// mod050-test.cpp
// Modern C++ test 50
// Test #750


#include <utility>
auto func() { return std::make_pair(50, 50+1); }
int main() { auto [a,b] = func(); return a; }
