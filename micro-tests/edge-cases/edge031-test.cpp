// edge031-test.cpp
// Edge case test 31
// Test #791


int func() { volatile int x = 31; return x; }
int main() { return func(); }
