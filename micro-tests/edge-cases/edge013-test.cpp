// edge013-test.cpp
// Edge case test 13
// Test #773


int func() { volatile int x = 13; return x; }
int main() { return func(); }
