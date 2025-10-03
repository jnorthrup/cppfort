// cls025-basic.cpp
// Basic class 25
// Test #425


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(25); return t.get(); }
