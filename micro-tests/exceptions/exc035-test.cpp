// exc035-test.cpp
// Exception test 35
// Test #695


int func() { try { throw 35; } catch(int e) { return e; } return 0; }
int main() { return func(); }
