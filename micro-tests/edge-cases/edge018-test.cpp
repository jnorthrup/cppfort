// edge018-test.cpp
// Edge case test 18
// Test #778


int func() { volatile int x = 18; return x; }
int main() { return func(); }
