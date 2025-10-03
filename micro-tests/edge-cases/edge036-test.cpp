// edge036-test.cpp
// Edge case test 36
// Test #796


int func() { volatile int x = 36; return x; }
int main() { return func(); }
