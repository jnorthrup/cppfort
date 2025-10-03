// exc025-test.cpp
// Exception test 25
// Test #685


int func() { try { throw 25; } catch(int e) { return e; } return 0; }
int main() { return func(); }
