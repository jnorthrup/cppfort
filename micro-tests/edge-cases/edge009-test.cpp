// edge009-test.cpp
// Edge case test 9
// Test #769


int func() { volatile int x = 9; return x; }
int main() { return func(); }
