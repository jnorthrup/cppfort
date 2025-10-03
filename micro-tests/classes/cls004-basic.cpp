// cls004-basic.cpp
// Basic class 4
// Test #404


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(4); return t.get(); }
