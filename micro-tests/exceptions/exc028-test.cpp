// exc028-test.cpp
// Exception test 28
// Test #688


int func() { try { throw 28; } catch(int e) { return e; } return 0; }
int main() { return func(); }
