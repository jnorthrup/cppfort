// exc017-test.cpp
// Exception test 17
// Test #677


int func() { try { throw 17; } catch(int e) { return e; } return 0; }
int main() { return func(); }
