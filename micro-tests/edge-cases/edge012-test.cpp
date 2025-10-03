// edge012-test.cpp
// Edge case test 12
// Test #772


int func() { volatile int x = 12; return x; }
int main() { return func(); }
