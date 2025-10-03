// exc016-test.cpp
// Exception test 16
// Test #676


int func() { try { throw 16; } catch(int e) { return e; } return 0; }
int main() { return func(); }
