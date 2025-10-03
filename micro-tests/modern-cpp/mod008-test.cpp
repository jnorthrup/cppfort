// mod008-test.cpp
// Modern C++ test 8
// Test #708


#include <utility>
auto func() { return std::make_pair(8, 8+1); }
int main() { auto [a,b] = func(); return a; }
