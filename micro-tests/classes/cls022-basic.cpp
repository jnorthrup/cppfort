// cls022-basic.cpp
// Basic class 22
// Test #422


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(22); return t.get(); }
