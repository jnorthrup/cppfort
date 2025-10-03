// exc004-test.cpp
// Exception test 4
// Test #664


int func() { try { throw 4; } catch(int e) { return e; } return 0; }
int main() { return func(); }
