// exc024-test.cpp
// Exception test 24
// Test #684


int func() { try { throw 24; } catch(int e) { return e; } return 0; }
int main() { return func(); }
