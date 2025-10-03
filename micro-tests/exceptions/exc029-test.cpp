// exc029-test.cpp
// Exception test 29
// Test #689


int func() { try { throw 29; } catch(int e) { return e; } return 0; }
int main() { return func(); }
