// edge005-test.cpp
// Edge case test 5
// Test #765


int func() { volatile int x = 5; return x; }
int main() { return func(); }
