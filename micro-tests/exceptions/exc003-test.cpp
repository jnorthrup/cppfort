// exc003-test.cpp
// Exception test 3
// Test #663


int func() { try { throw 3; } catch(int e) { return e; } return 0; }
int main() { return func(); }
