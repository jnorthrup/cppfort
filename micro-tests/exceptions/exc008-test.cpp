// exc008-test.cpp
// Exception test 8
// Test #668


int func() { try { throw 8; } catch(int e) { return e; } return 0; }
int main() { return func(); }
