// exc037-test.cpp
// Exception test 37
// Test #697


int func() { try { throw 37; } catch(int e) { return e; } return 0; }
int main() { return func(); }
