// exc023-test.cpp
// Exception test 23
// Test #683


int func() { try { throw 23; } catch(int e) { return e; } return 0; }
int main() { return func(); }
