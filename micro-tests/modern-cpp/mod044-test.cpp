// mod044-test.cpp
// Modern C++ test 44
// Test #744


#include <utility>
auto func() { return std::make_pair(44, 44+1); }
int main() { auto [a,b] = func(); return a; }
