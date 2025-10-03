// exc034-test.cpp
// Exception test 34
// Test #694


int func() { try { throw 34; } catch(int e) { return e; } return 0; }
int main() { return func(); }
