// exc021-test.cpp
// Exception test 21
// Test #681


int func() { try { throw 21; } catch(int e) { return e; } return 0; }
int main() { return func(); }
