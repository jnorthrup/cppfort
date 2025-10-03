// mod054-test.cpp
// Modern C++ test 54
// Test #754


#include <utility>
auto func() { return std::make_pair(54, 54+1); }
int main() { auto [a,b] = func(); return a; }
