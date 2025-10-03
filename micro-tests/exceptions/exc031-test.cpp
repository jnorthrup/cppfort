// exc031-test.cpp
// Exception test 31
// Test #691


int func() { try { throw 31; } catch(int e) { return e; } return 0; }
int main() { return func(); }
