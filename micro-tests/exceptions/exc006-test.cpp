// exc006-test.cpp
// Exception test 6
// Test #666


int func() { try { throw 6; } catch(int e) { return e; } return 0; }
int main() { return func(); }
