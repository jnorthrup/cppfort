// mod035-test.cpp
// Modern C++ test 35
// Test #735


#include <utility>
auto func() { return std::make_pair(35, 35+1); }
int main() { auto [a,b] = func(); return a; }
