// edge020-test.cpp
// Edge case test 20
// Test #780


int func() { volatile int x = 20; return x; }
int main() { return func(); }
