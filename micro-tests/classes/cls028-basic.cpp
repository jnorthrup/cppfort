// cls028-basic.cpp
// Basic class 28
// Test #428


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(28); return t.get(); }
