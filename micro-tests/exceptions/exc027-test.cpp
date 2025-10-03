// exc027-test.cpp
// Exception test 27
// Test #687


int func() { try { throw 27; } catch(int e) { return e; } return 0; }
int main() { return func(); }
