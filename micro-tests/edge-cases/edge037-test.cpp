// edge037-test.cpp
// Edge case test 37
// Test #797


int func() { volatile int x = 37; return x; }
int main() { return func(); }
