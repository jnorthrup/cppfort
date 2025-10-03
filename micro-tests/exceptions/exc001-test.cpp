// exc001-test.cpp
// Exception test 1
// Test #661


int func() { try { throw 1; } catch(int e) { return e; } return 0; }
int main() { return func(); }
