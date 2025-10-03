// edge008-test.cpp
// Edge case test 8
// Test #768


int func() { volatile int x = 8; return x; }
int main() { return func(); }
