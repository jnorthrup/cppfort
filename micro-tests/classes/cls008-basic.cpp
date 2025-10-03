// cls008-basic.cpp
// Basic class 8
// Test #408


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(8); return t.get(); }
