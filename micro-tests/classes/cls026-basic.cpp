// cls026-basic.cpp
// Basic class 26
// Test #426


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(26); return t.get(); }
