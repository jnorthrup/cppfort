// exc033-test.cpp
// Exception test 33
// Test #693


int func() { try { throw 33; } catch(int e) { return e; } return 0; }
int main() { return func(); }
