// mod017-test.cpp
// Modern C++ test 17
// Test #717


#include <utility>
auto func() { return std::make_pair(17, 17+1); }
int main() { auto [a,b] = func(); return a; }
