// exc002-test.cpp
// Exception test 2
// Test #662


int func() { try { throw 2; } catch(int e) { return e; } return 0; }
int main() { return func(); }
