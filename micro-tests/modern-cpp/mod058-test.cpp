// mod058-test.cpp
// Modern C++ test 58
// Test #758


#include <utility>
auto func() { return std::make_pair(58, 58+1); }
int main() { auto [a,b] = func(); return a; }
