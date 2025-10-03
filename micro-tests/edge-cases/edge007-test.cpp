// edge007-test.cpp
// Edge case test 7
// Test #767


int func() { volatile int x = 7; return x; }
int main() { return func(); }
