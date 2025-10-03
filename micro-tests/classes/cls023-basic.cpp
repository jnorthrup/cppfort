// cls023-basic.cpp
// Basic class 23
// Test #423


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(23); return t.get(); }
