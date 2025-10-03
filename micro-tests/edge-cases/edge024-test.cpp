// edge024-test.cpp
// Edge case test 24
// Test #784


int func() { volatile int x = 24; return x; }
int main() { return func(); }
