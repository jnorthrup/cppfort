// cls002-basic.cpp
// Basic class 2
// Test #402


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(2); return t.get(); }
