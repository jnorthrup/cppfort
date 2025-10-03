// mod021-test.cpp
// Modern C++ test 21
// Test #721


#include <utility>
auto func() { return std::make_pair(21, 21+1); }
int main() { auto [a,b] = func(); return a; }
