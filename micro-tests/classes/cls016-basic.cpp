// cls016-basic.cpp
// Basic class 16
// Test #416


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(16); return t.get(); }
