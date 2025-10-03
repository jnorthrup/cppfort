// edge017-test.cpp
// Edge case test 17
// Test #777


int func() { volatile int x = 17; return x; }
int main() { return func(); }
