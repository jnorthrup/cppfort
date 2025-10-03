// edge027-test.cpp
// Edge case test 27
// Test #787


int func() { volatile int x = 27; return x; }
int main() { return func(); }
