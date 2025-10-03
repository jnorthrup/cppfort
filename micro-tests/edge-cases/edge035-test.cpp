// edge035-test.cpp
// Edge case test 35
// Test #795


int func() { volatile int x = 35; return x; }
int main() { return func(); }
