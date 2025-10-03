// exc014-test.cpp
// Exception test 14
// Test #674


int func() { try { throw 14; } catch(int e) { return e; } return 0; }
int main() { return func(); }
