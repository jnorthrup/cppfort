// cls005-basic.cpp
// Basic class 5
// Test #405


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(5); return t.get(); }
