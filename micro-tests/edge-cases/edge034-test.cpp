// edge034-test.cpp
// Edge case test 34
// Test #794


int func() { volatile int x = 34; return x; }
int main() { return func(); }
