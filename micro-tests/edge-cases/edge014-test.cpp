// edge014-test.cpp
// Edge case test 14
// Test #774


int func() { volatile int x = 14; return x; }
int main() { return func(); }
