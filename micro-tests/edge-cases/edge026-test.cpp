// edge026-test.cpp
// Edge case test 26
// Test #786


int func() { volatile int x = 26; return x; }
int main() { return func(); }
