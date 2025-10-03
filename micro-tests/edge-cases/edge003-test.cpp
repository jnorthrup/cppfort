// edge003-test.cpp
// Edge case test 3
// Test #763


int func() { volatile int x = 3; return x; }
int main() { return func(); }
