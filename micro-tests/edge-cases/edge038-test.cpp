// edge038-test.cpp
// Edge case test 38
// Test #798


int func() { volatile int x = 38; return x; }
int main() { return func(); }
