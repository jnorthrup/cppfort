// exc018-test.cpp
// Exception test 18
// Test #678


int func() { try { throw 18; } catch(int e) { return e; } return 0; }
int main() { return func(); }
