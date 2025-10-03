// edge029-test.cpp
// Edge case test 29
// Test #789


int func() { volatile int x = 29; return x; }
int main() { return func(); }
