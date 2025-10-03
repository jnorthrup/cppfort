// cls006-basic.cpp
// Basic class 6
// Test #406


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(6); return t.get(); }
