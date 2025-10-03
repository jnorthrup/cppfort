// exc005-test.cpp
// Exception test 5
// Test #665


int func() { try { throw 5; } catch(int e) { return e; } return 0; }
int main() { return func(); }
