// exc011-test.cpp
// Exception test 11
// Test #671


int func() { try { throw 11; } catch(int e) { return e; } return 0; }
int main() { return func(); }
