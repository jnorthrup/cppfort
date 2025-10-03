// cls029-basic.cpp
// Basic class 29
// Test #429


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(29); return t.get(); }
