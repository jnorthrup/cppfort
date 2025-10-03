// edge015-test.cpp
// Edge case test 15
// Test #775


int func() { volatile int x = 15; return x; }
int main() { return func(); }
