// edge032-test.cpp
// Edge case test 32
// Test #792


int func() { volatile int x = 32; return x; }
int main() { return func(); }
