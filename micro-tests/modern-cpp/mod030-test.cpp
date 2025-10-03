// mod030-test.cpp
// Modern C++ test 30
// Test #730


#include <utility>
auto func() { return std::make_pair(30, 30+1); }
int main() { auto [a,b] = func(); return a; }
