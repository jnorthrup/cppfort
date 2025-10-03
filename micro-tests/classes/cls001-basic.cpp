// cls001-basic.cpp
// Basic class 1
// Test #401


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(1); return t.get(); }
