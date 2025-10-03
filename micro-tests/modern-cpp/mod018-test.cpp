// mod018-test.cpp
// Modern C++ test 18
// Test #718


#include <utility>
auto func() { return std::make_pair(18, 18+1); }
int main() { auto [a,b] = func(); return a; }
