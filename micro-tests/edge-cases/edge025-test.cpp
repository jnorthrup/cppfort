// edge025-test.cpp
// Edge case test 25
// Test #785


int func() { volatile int x = 25; return x; }
int main() { return func(); }
