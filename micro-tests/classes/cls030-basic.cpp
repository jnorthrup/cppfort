// cls030-basic.cpp
// Basic class 30
// Test #430


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(30); return t.get(); }
