// edge016-test.cpp
// Edge case test 16
// Test #776


int func() { volatile int x = 16; return x; }
int main() { return func(); }
