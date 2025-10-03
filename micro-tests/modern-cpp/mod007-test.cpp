// mod007-test.cpp
// Modern C++ test 7
// Test #707


#include <utility>
auto func() { return std::make_pair(7, 7+1); }
int main() { auto [a,b] = func(); return a; }
