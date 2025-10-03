// edge022-test.cpp
// Edge case test 22
// Test #782


int func() { volatile int x = 22; return x; }
int main() { return func(); }
