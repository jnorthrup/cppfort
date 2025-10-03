// exc040-test.cpp
// Exception test 40
// Test #700


int func() { try { throw 40; } catch(int e) { return e; } return 0; }
int main() { return func(); }
