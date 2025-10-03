// exc022-test.cpp
// Exception test 22
// Test #682


int func() { try { throw 22; } catch(int e) { return e; } return 0; }
int main() { return func(); }
