// mod022-test.cpp
// Modern C++ test 22
// Test #722


#include <utility>
auto func() { return std::make_pair(22, 22+1); }
int main() { auto [a,b] = func(); return a; }
