// edge023-test.cpp
// Edge case test 23
// Test #783


int func() { volatile int x = 23; return x; }
int main() { return func(); }
