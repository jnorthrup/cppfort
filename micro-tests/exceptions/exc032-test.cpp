// exc032-test.cpp
// Exception test 32
// Test #692


int func() { try { throw 32; } catch(int e) { return e; } return 0; }
int main() { return func(); }
