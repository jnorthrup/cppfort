// cls024-basic.cpp
// Basic class 24
// Test #424


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(24); return t.get(); }
