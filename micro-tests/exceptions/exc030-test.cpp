// exc030-test.cpp
// Exception test 30
// Test #690


int func() { try { throw 30; } catch(int e) { return e; } return 0; }
int main() { return func(); }
