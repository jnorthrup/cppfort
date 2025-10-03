// exc015-test.cpp
// Exception test 15
// Test #675


int func() { try { throw 15; } catch(int e) { return e; } return 0; }
int main() { return func(); }
