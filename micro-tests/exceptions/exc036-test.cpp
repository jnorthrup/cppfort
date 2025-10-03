// exc036-test.cpp
// Exception test 36
// Test #696


int func() { try { throw 36; } catch(int e) { return e; } return 0; }
int main() { return func(); }
