// cls007-basic.cpp
// Basic class 7
// Test #407


class Test{ int x; public: Test(int v):x(v){} int get(){return x;} };
int main(){ Test t(7); return t.get(); }
