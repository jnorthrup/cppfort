// cls011-basic.cpp
// Basic class 11
// Test #411


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(11); return t.get(); }
