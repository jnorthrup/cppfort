// exc007-test.cpp
// Exception test 7
// Test #667


int func() { try { throw 7; } catch(int e) { return e; } return 0; }
int main() { return func(); }
