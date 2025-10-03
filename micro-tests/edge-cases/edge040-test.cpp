// edge040-test.cpp
// Edge case test 40
// Test #800


int func() { volatile int x = 40; return x; }
int main() { return func(); }
