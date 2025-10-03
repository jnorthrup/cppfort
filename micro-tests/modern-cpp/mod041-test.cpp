// mod041-test.cpp
// Modern C++ test 41
// Test #741


#include <utility>
auto func() { return std::make_pair(41, 41+1); }
int main() { auto [a,b] = func(); return a; }
