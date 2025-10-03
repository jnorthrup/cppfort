// exc038-test.cpp
// Exception test 38
// Test #698


int func() { try { throw 38; } catch(int e) { return e; } return 0; }
int main() { return func(); }
