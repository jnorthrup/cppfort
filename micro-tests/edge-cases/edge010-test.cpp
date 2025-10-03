// edge010-test.cpp
// Edge case test 10
// Test #770


int func() { volatile int x = 10; return x; }
int main() { return func(); }
