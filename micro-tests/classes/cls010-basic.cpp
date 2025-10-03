// cls010-basic.cpp
// Basic class 10
// Test #410


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(10); return t.get(); }
