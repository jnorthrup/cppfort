// exc020-test.cpp
// Exception test 20
// Test #680


int func() { try { throw 20; } catch(int e) { return e; } return 0; }
int main() { return func(); }
