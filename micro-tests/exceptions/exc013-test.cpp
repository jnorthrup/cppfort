// exc013-test.cpp
// Exception test 13
// Test #673


int func() { try { throw 13; } catch(int e) { return e; } return 0; }
int main() { return func(); }
