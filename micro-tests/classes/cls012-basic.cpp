// cls012-basic.cpp
// Basic class 12
// Test #412


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(12); return t.get(); }
