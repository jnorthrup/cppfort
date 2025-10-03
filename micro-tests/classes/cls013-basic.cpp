// cls013-basic.cpp
// Basic class 13
// Test #413


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(13); return t.get(); }
