// exc009-test.cpp
// Exception test 9
// Test #669


int func() { try { throw 9; } catch(int e) { return e; } return 0; }
int main() { return func(); }
