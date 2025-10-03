// exc012-test.cpp
// Exception test 12
// Test #672


int func() { try { throw 12; } catch(int e) { return e; } return 0; }
int main() { return func(); }
