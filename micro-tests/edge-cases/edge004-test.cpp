// edge004-test.cpp
// Edge case test 4
// Test #764


int func() { volatile int x = 4; return x; }
int main() { return func(); }
