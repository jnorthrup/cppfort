// exc010-test.cpp
// Exception test 10
// Test #670


int func() { try { throw 10; } catch(int e) { return e; } return 0; }
int main() { return func(); }
