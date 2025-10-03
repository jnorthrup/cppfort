// edge019-test.cpp
// Edge case test 19
// Test #779


int func() { volatile int x = 19; return x; }
int main() { return func(); }
