// mod005-test.cpp
// Modern C++ test 5
// Test #705


#include <utility>
auto func() { return std::make_pair(5, 5+1); }
int main() { auto [a,b] = func(); return a; }
