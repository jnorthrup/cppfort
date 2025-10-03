// cls014-basic.cpp
// Basic class 14
// Test #414


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(14); return t.get(); }
