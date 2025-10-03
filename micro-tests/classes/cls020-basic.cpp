// cls020-basic.cpp
// Basic class 20
// Test #420


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(20); return t.get(); }
