// edge039-test.cpp
// Edge case test 39
// Test #799


int func() { volatile int x = 39; return x; }
int main() { return func(); }
