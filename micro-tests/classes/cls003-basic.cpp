// cls003-basic.cpp
// Basic class 3
// Test #403


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(3); return t.get(); }
