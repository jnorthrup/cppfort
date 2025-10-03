// edge021-test.cpp
// Edge case test 21
// Test #781


int func() { volatile int x = 21; return x; }
int main() { return func(); }
