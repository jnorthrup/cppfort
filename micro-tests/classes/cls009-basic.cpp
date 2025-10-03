// cls009-basic.cpp
// Basic class 9
// Test #409


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(9); return t.get(); }
