// exc039-test.cpp
// Exception test 39
// Test #699


int func() { try { throw 39; } catch(int e) { return e; } return 0; }
int main() { return func(); }
