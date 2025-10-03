// edge002-test.cpp
// Edge case test 2
// Test #762


int func() { volatile int x = 2; return x; }
int main() { return func(); }
