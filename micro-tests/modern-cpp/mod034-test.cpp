// mod034-test.cpp
// Modern C++ test 34
// Test #734


#include <utility>
auto func() { return std::make_pair(34, 34+1); }
int main() { auto [a,b] = func(); return a; }
