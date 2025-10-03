// edge006-test.cpp
// Edge case test 6
// Test #766


int func() { volatile int x = 6; return x; }
int main() { return func(); }
