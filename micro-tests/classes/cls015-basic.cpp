// cls015-basic.cpp
// Basic class 15
// Test #415


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(15); return t.get(); }
