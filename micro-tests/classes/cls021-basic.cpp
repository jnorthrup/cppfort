// cls021-basic.cpp
// Basic class 21
// Test #421


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(21); return t.get(); }
