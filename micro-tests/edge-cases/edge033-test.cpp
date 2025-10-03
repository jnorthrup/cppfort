// edge033-test.cpp
// Edge case test 33
// Test #793


int func() { volatile int x = 33; return x; }
int main() { return func(); }
