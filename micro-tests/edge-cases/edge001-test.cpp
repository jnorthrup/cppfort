// edge001-test.cpp
// Edge case test 1
// Test #761


int func() { volatile int x = 1; return x; }
int main() { return func(); }
