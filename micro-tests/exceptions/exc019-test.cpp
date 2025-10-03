// exc019-test.cpp
// Exception test 19
// Test #679


int func() { try { throw 19; } catch(int e) { return e; } return 0; }
int main() { return func(); }
