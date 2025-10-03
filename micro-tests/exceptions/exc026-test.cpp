// exc026-test.cpp
// Exception test 26
// Test #686


int func() { try { throw 26; } catch(int e) { return e; } return 0; }
int main() { return func(); }
