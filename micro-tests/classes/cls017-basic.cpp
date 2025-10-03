// cls017-basic.cpp
// Basic class 17
// Test #417


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(17); return t.get(); }
