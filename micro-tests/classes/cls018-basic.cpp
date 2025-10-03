// cls018-basic.cpp
// Basic class 18
// Test #418


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(18); return t.get(); }
