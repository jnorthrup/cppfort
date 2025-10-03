// edge030-test.cpp
// Edge case test 30
// Test #790


int func() { volatile int x = 30; return x; }
int main() { return func(); }
