// mod057-test.cpp
// Modern C++ test 57
// Test #757


#include <utility>
auto func() { return std::make_pair(57, 57+1); }
int main() { auto [a,b] = func(); return a; }
