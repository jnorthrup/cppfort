// edge028-test.cpp
// Edge case test 28
// Test #788


int func() { volatile int x = 28; return x; }
int main() { return func(); }
