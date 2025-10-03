// cls019-basic.cpp
// Basic class 19
// Test #419


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(19); return t.get(); }
