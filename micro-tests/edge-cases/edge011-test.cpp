// edge011-test.cpp
// Edge case test 11
// Test #771


int func() { volatile int x = 11; return x; }
int main() { return func(); }
